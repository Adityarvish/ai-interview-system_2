"""
semantic_evaluator.py — 9-Stage Semantic Evaluation Engine

Architecture
============
Stage 1  : Transcript Cleaning               (sequential — input to all others)
Stages 2–6: Run in parallel via asyncio.gather()
  Stage 2  : Semantic Concept Coverage        (Embeddings + Cosine Similarity)
  Stage 3  : Technical Accuracy               (LLM rubric scoring)
  Stage 4  : Completeness                     (concept gap analysis)
  Stage 5  : Communication                    (clarity, structure, filler)
  Stage 6  : Problem Solving                  (STAR-lite + reasoning depth)
Stage 7  : STAR Evaluation                   (behavioral only, sequential gate)
Stage 8  : Score Aggregation                 (weighted formula)
Stage 9  : Confidence Score                  (answer evidence + model certainty)

Output per question
-------------------
  question_score   : int 0–100
  skill_score      : int 0–100  (domain-specific)
  strengths        : List[str]
  weaknesses       : List[str]
  confidence_score : float 0–1

Design principles
-----------------
- All public methods are async.
- CPU-bound operations (embedding, cosine sim) run in a thread pool via
  asyncio.run_in_executor so the event loop is never blocked.
- Stages 2–6 run concurrently via asyncio.gather(); total latency equals
  the slowest stage, not the sum.
- Evaluation is triggered as a BackgroundTask in the router, returning the
  next question immediately while scoring runs behind the scenes.
- LLM calls use Groq (existing llm_service) via run_in_executor.
- Embeddings use sentence-transformers (already in requirements).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ─── lazy-import helpers so startup stays fast ─────────────────────────────

@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load sentence-transformer once, cached for the process lifetime."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("[SEM-EVAL] Embedding model loaded (all-MiniLM-L6-v2)")
    return model


# ─── data classes ──────────────────────────────────────────────────────────

@dataclass
class StageResult:
    stage_name: str
    score: float           # 0–100
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0


@dataclass
class QuestionEvalResult:
    question_id: int
    interview_id: str
    question: str
    transcript_raw: str
    transcript_clean: str

    # Per-stage results
    concept_coverage: StageResult = field(default_factory=lambda: StageResult("concept_coverage", 0))
    technical_accuracy: StageResult = field(default_factory=lambda: StageResult("technical_accuracy", 0))
    completeness: StageResult = field(default_factory=lambda: StageResult("completeness", 0))
    communication: StageResult = field(default_factory=lambda: StageResult("communication", 0))
    problem_solving: StageResult = field(default_factory=lambda: StageResult("problem_solving", 0))
    star_evaluation: Optional[StageResult] = None  # behavioral only

    # Aggregated outputs
    question_score: int = 0
    skill_score: int = 0
    confidence_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    total_elapsed_ms: int = 0
    is_behavioral: bool = False


# ─── Stage 1 — Transcript Cleaning ─────────────────────────────────────────

class TranscriptCleaner:
    """
    Stage 1: Clean raw STT transcript before semantic analysis.

    Operations (all regex, no LLM — must be near-zero latency):
      • Remove filler words (um, uh, like, you know, so, right, basically…)
      • Strip disfluencies and repeated words
      • Normalise whitespace and punctuation
      • Expand common technical contractions (I've → I have, etc.)
      • Preserve technical terms and acronyms (case-sensitive guard)
    """

    _FILLERS = re.compile(
        r'\b(um+|uh+|uhh+|er+|hmm+|like(?!\s+a\s+\w)|you know|you know what i mean|'
        r'basically|literally|actually|honestly|right\?|okay\?|so+\s|kind of|sort of|'
        r'i mean|i guess|i think that|let me think|well\s+so|right so)\b',
        re.IGNORECASE
    )
    _REPEAT_WORDS = re.compile(r'\b(\w+)(\s+\1){2,}\b', re.IGNORECASE)
    _MULTI_SPACE  = re.compile(r' {2,}')
    _CONTRACTIONS = {
        "i've": "I have", "i'm": "I am", "i'd": "I would", "i'll": "I will",
        "it's": "it is", "that's": "that is", "we've": "we have",
        "we're": "we are", "they're": "they are", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "can't": "cannot",
        "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
        "there's": "there is", "here's": "here is", "let's": "let us",
    }

    @classmethod
    def clean(cls, raw: str) -> str:
        if not raw:
            return ""
        text = raw.strip()

        # Expand contractions
        for contraction, expanded in cls._CONTRACTIONS.items():
            text = re.sub(r'\b' + re.escape(contraction) + r'\b', expanded, text, flags=re.IGNORECASE)

        # Remove filler words
        text = cls._FILLERS.sub(' ', text)

        # Collapse repeated words (stutters): "the the the" → "the"
        text = cls._REPEAT_WORDS.sub(r'\1', text)

        # Normalise whitespace
        text = cls._MULTI_SPACE.sub(' ', text).strip()

        # Ensure sentence endings
        if text and text[-1] not in '.!?':
            text += '.'

        return text


# ─── Stage 2 — Semantic Concept Coverage ───────────────────────────────────

class ConceptCoverageStage:
    """
    Stage 2: Measure how many expected concepts the answer covers.

    Method:
      1. Generate expected concept embeddings from the question + JD keywords.
      2. Embed the cleaned answer.
      3. Cosine similarity between answer embedding and each concept embedding.
      4. Coverage = fraction of concepts with similarity ≥ COVERAGE_THRESHOLD.
      5. Score = coverage × 100, adjusted by answer length.

    All numpy/sentence-transformer work runs in the thread pool.
    """

    COVERAGE_THRESHOLD = 0.35   # cosine sim to count a concept as "covered"
    MIN_ANSWER_WORDS   = 20     # below this → heavy penalty
    IDEAL_ANSWER_WORDS = 80     # above this → full length bonus

    async def evaluate(
        self,
        question: str,
        answer_clean: str,
        job_description: str,
        resume_context: str,
        loop: asyncio.AbstractEventLoop,
    ) -> StageResult:
        t0 = time.perf_counter()

        def _compute():
            model = _get_embedding_model()

            concepts = self._extract_concepts(question, job_description, resume_context)
            if not concepts:
                return StageResult("concept_coverage", 50, {"note": "no concepts extracted"})

            concept_embeddings = model.encode(concepts, normalize_embeddings=True)
            answer_embedding   = model.encode([answer_clean], normalize_embeddings=True)[0]

            sims     = np.dot(concept_embeddings, answer_embedding)
            covered  = int(np.sum(sims >= self.COVERAGE_THRESHOLD))
            coverage = covered / len(concepts)

            # Length adjustment
            word_count   = len(answer_clean.split())
            length_factor = min(1.0, word_count / self.IDEAL_ANSWER_WORDS)
            if word_count < self.MIN_ANSWER_WORDS:
                length_factor *= 0.6

            score = min(100, int(coverage * 100 * (0.7 + 0.3 * length_factor)))

            return StageResult(
                "concept_coverage",
                score,
                {
                    "concepts_checked":  len(concepts),
                    "concepts_covered":  covered,
                    "coverage_ratio":    round(coverage, 3),
                    "top_similarities":  [round(float(s), 3) for s in sorted(sims, reverse=True)[:5]],
                    "word_count":        word_count,
                    "length_factor":     round(length_factor, 3),
                    "concepts":          concepts[:8],
                },
            )

        result = await loop.run_in_executor(None, _compute)
        result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE2] concept_coverage={result.score} in {result.elapsed_ms}ms")
        return result

    @staticmethod
    def _extract_concepts(question: str, jd: str, resume: str) -> List[str]:
        """
        Extract key concepts to check coverage against.
        Combines noun phrases from question + top JD/resume keywords.
        No LLM — pure heuristic for speed.
        """
        # Pull meaningful tokens: 3+ char, not stop words
        STOPS = {
            'the','a','an','and','or','but','in','on','at','to','for',
            'of','with','by','from','is','are','was','were','be','been',
            'have','has','had','do','does','did','will','would','could',
            'should','may','might','can','about','your','you','how','what',
            'when','where','why','which','who','this','that','these','those',
            'tell','describe','explain','discuss','walk','through','give',
            'example','time','please','any','some','most','more','less',
        }

        def keywords(text: str, limit: int) -> List[str]:
            tokens = re.findall(r'\b[A-Za-z][a-z]{2,}\b', text)
            seen, out = set(), []
            for t in tokens:
                tl = t.lower()
                if tl not in STOPS and tl not in seen:
                    seen.add(tl)
                    out.append(t)
                    if len(out) >= limit:
                        break
            return out

        q_kws  = keywords(question, 6)
        jd_kws = keywords(jd, 10)
        r_kws  = keywords(resume, 8)

        # Also extract explicit skill/tool mentions (CamelCase, ALL_CAPS, version nums)
        tech_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9]+|[A-Z]{2,}[0-9]*)\b')
        tech_terms   = list(dict.fromkeys(
            t for t in tech_pattern.findall(question + " " + jd)
            if len(t) > 1 and t not in ('I', 'A', 'The', 'In', 'At', 'We')
        ))[:6]

        all_concepts = list(dict.fromkeys(q_kws + jd_kws[:6] + r_kws[:4] + tech_terms))
        return all_concepts[:20]


# ─── Stage 3 — Technical Accuracy ──────────────────────────────────────────

class TechnicalAccuracyStage:
    """
    Stage 3: Use the LLM to score factual/technical correctness.

    Rubric (0–100):
      90–100 : All technical claims accurate; specific, correct terminology
      70–89  : Mostly accurate with minor imprecisions
      50–69  : Partially correct; some misconceptions or surface-level
      30–49  : Significant inaccuracies or vague hand-waving
      0–29   : Mostly wrong or no technical substance
    """

    async def evaluate(
        self,
        question: str,
        answer_clean: str,
        job_description: str,
        llm_fn,          # callable: (prompt: str) -> str  (sync, runs in executor)
        loop: asyncio.AbstractEventLoop,
    ) -> StageResult:
        t0 = time.perf_counter()

        prompt = self._build_prompt(question, answer_clean, job_description)

        def _call_llm():
            try:
                return llm_fn(prompt)
            except Exception as e:
                logger.warning(f"[STAGE3] LLM call failed: {e}")
                return ""

        raw = await loop.run_in_executor(None, _call_llm)
        result = self._parse(raw)
        result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE3] technical_accuracy={result.score} in {result.elapsed_ms}ms")
        return result

    @staticmethod
    def _build_prompt(question: str, answer: str, jd: str) -> str:
        return f"""You are a senior technical interviewer. Score the technical accuracy of this answer.

QUESTION: {question[:300]}
ANSWER: {answer[:600]}
ROLE CONTEXT: {jd[:200]}

Score ONLY technical accuracy (factual correctness, proper terminology, valid reasoning).
Ignore communication style — focus purely on whether the technical content is correct.

Respond ONLY with this JSON (no markdown, no preamble):
{{"score": <integer 0-100>, "accurate_claims": ["<claim1>", "<claim2>"], "inaccurate_claims": ["<error1>"], "reasoning": "<one sentence>"}}"""

    @staticmethod
    def _parse(raw: str) -> StageResult:
        import json
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                d = json.loads(m.group(0))
                score     = max(0, min(100, int(d.get("score", 50))))
                accurate  = d.get("accurate_claims", [])[:4]
                inaccurate= d.get("inaccurate_claims", [])[:3]
                reasoning = d.get("reasoning", "")
                return StageResult("technical_accuracy", score, {
                    "accurate_claims":   accurate,
                    "inaccurate_claims": inaccurate,
                    "reasoning":         reasoning,
                })
        except Exception:
            pass
        return StageResult("technical_accuracy", 50, {"note": "parse_failed"})


# ─── Stage 4 — Completeness ─────────────────────────────────────────────────

class CompletenessStage:
    """
    Stage 4: Measure answer completeness — did the candidate address all parts
    of the question?

    Approach:
      1. Split question into sub-questions / required elements (heuristic).
      2. Check each sub-question against the answer using embedding similarity.
      3. Score = (addressed sub-questions / total) × 100.

    Fast path: if the question has no discernible sub-parts, fall back to
    word-count + concept-coverage proxy.
    """

    ADDRESSED_THRESHOLD = 0.32

    async def evaluate(
        self,
        question: str,
        answer_clean: str,
        loop: asyncio.AbstractEventLoop,
    ) -> StageResult:
        t0 = time.perf_counter()

        def _compute():
            sub_questions = self._split_question(question)
            if len(sub_questions) <= 1:
                # Single-part question — use word count proxy
                wc = len(answer_clean.split())
                score = min(100, max(20, int(wc / 1.2)))
                return StageResult("completeness", score, {
                    "mode":        "word_count_proxy",
                    "word_count":  wc,
                })

            model  = _get_embedding_model()
            sq_emb = model.encode(sub_questions, normalize_embeddings=True)
            ans_emb= model.encode([answer_clean], normalize_embeddings=True)[0]
            sims   = np.dot(sq_emb, ans_emb)

            addressed = [sq for sq, s in zip(sub_questions, sims)
                         if s >= self.ADDRESSED_THRESHOLD]
            ratio   = len(addressed) / len(sub_questions)
            score   = int(ratio * 100)

            gaps = [sq for sq, s in zip(sub_questions, sims)
                    if s < self.ADDRESSED_THRESHOLD]

            return StageResult("completeness", score, {
                "sub_questions":     sub_questions,
                "addressed":         addressed,
                "gaps":              gaps,
                "completeness_ratio": round(ratio, 3),
            })

        result = await loop.run_in_executor(None, _compute)
        result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE4] completeness={result.score} in {result.elapsed_ms}ms")
        return result

    @staticmethod
    def _split_question(question: str) -> List[str]:
        """
        Heuristic: split multi-part question into atomic sub-questions.
        Handles:
          • "Tell me about X, what Y, and how Z"
          • "Describe X. Why did you Y? How did you Z?"
          • Single-part questions → list of 1
        """
        # Split on sentence boundaries first
        sentences = re.split(r'(?<=[.?!])\s+', question.strip())
        parts = []
        for sent in sentences:
            # Split on conjunctions: "X and Y", "X, and also Z"
            sub = re.split(r',?\s+(?:and|also|additionally|furthermore|moreover)\s+', sent, flags=re.IGNORECASE)
            parts.extend([s.strip() for s in sub if len(s.strip()) > 15])

        # Deduplicate while preserving order
        seen, unique = set(), []
        for p in parts:
            key = p.lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique if unique else [question]


# ─── Stage 5 — Communication ────────────────────────────────────────────────

class CommunicationStage:
    """
    Stage 5: Score clarity, structure, and communication quality.

    Sub-scores (weighted average):
      • Clarity        (40%) — sentence length variance, reading ease proxy
      • Structure      (30%) — logical connectives, example signalling
      • Conciseness    (20%) — filler ratio (pre-cleaning fillers, if any)
      • Specificity    (10%) — concrete nouns, numbers, named examples

    Fully heuristic — no LLM, near-zero latency.
    """

    async def evaluate(
        self,
        transcript_raw: str,
        transcript_clean: str,
        loop: asyncio.AbstractEventLoop,
    ) -> StageResult:
        t0 = time.perf_counter()

        def _compute():
            clarity      = self._clarity_score(transcript_clean)
            structure    = self._structure_score(transcript_clean)
            conciseness  = self._conciseness_score(transcript_raw, transcript_clean)
            specificity  = self._specificity_score(transcript_clean)

            weighted = (
                clarity     * 0.40 +
                structure   * 0.30 +
                conciseness * 0.20 +
                specificity * 0.10
            )
            score = int(min(100, max(0, weighted)))

            return StageResult("communication", score, {
                "clarity":      round(clarity, 1),
                "structure":    round(structure, 1),
                "conciseness":  round(conciseness, 1),
                "specificity":  round(specificity, 1),
                "word_count":   len(transcript_clean.split()),
            })

        result = await loop.run_in_executor(None, _compute)
        result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE5] communication={result.score} in {result.elapsed_ms}ms")
        return result

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]

    def _clarity_score(self, text: str) -> float:
        sentences = self._sentences(text)
        if not sentences:
            return 40.0
        lengths = [len(s.split()) for s in sentences]
        avg     = sum(lengths) / len(lengths)
        # Ideal: 12–22 words per sentence
        if 12 <= avg <= 22:
            base = 90.0
        elif 8 <= avg < 12 or 22 < avg <= 30:
            base = 72.0
        elif avg < 8:
            base = 55.0  # too terse
        else:
            base = 50.0  # run-ons

        # Penalise high variance (rambling vs. uniform)
        if len(lengths) > 1:
            variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
            penalty  = min(15, variance ** 0.5)
            base    -= penalty
        return max(0, base)

    @staticmethod
    def _structure_score(text: str) -> float:
        connectives  = len(re.findall(
            r'\b(first|second|third|finally|then|next|also|however|because|'
            r'therefore|so|as a result|for example|for instance|such as|'
            r'in addition|on the other hand|to summarize|in conclusion)\b',
            text, re.IGNORECASE))
        examples     = len(re.findall(r'\b(for example|for instance|such as|like when|specifically)\b', text, re.IGNORECASE))
        words        = len(text.split())
        conn_density = connectives / max(1, words / 100)
        score = 40 + min(40, conn_density * 20) + min(20, examples * 10)
        return min(100, score)

    @staticmethod
    def _conciseness_score(raw: str, clean: str) -> float:
        """Measure how much cleaner the transcript is vs raw — proxy for filler density."""
        raw_words   = len(raw.split())
        clean_words = len(clean.split())
        if raw_words == 0:
            return 50.0
        reduction = (raw_words - clean_words) / raw_words
        # 0–5% filler: excellent; 5–15%: good; 15–30%: average; >30%: poor
        if reduction <= 0.05:
            return 95.0
        elif reduction <= 0.15:
            return 80.0
        elif reduction <= 0.30:
            return 60.0
        else:
            return max(20, 60 - (reduction - 0.30) * 200)

    @staticmethod
    def _specificity_score(text: str) -> float:
        # Named entities: CamelCase words, numbers, version refs, quoted tools
        specifics = re.findall(r'\b([A-Z][a-z]+[A-Z]\w*|\d[\d.]+[a-zA-Z]*|"[^"]{2,30}")\b', text)
        count     = len(specifics)
        if count >= 4:
            return 95.0
        elif count == 3:
            return 80.0
        elif count == 2:
            return 65.0
        elif count == 1:
            return 50.0
        return 30.0


# ─── Stage 6 — Problem Solving ──────────────────────────────────────────────

class ProblemSolvingStage:
    """
    Stage 6: Assess systematic reasoning, depth, and trade-off awareness.

    Dimensions (heuristic + embedding):
      • Reasoning depth   — presence of causal chains ("because", "therefore")
      • Trade-off awareness — "but", "however", "trade-off", "alternatively"
      • Systematic approach — "first", "step", "then", "approach", "decided"
      • Evidence of outcome — "result", "improved", "achieved", "reduced"
    """

    async def evaluate(
        self,
        question: str,
        answer_clean: str,
        loop: asyncio.AbstractEventLoop,
    ) -> StageResult:
        t0 = time.perf_counter()

        def _compute():
            text  = answer_clean.lower()
            words = len(text.split())

            reasoning = self._pattern_density(text, [
                'because', 'therefore', 'since', 'as a result', 'which means',
                'led to', 'caused', 'so that', 'in order to', 'the reason'
            ], words, ideal_per_100=2.0)

            tradeoffs = self._pattern_density(text, [
                'however', 'but', 'on the other hand', 'trade-off', 'alternatively',
                'instead', 'rather than', 'although', 'despite', 'even though',
                'downside', 'limitation', 'drawback', 'constraint'
            ], words, ideal_per_100=1.5)

            systematic = self._pattern_density(text, [
                'first', 'second', 'step', 'approach', 'decided', 'considered',
                'analysed', 'analyzed', 'plan', 'strategy', 'method', 'process'
            ], words, ideal_per_100=2.5)

            outcome = self._pattern_density(text, [
                'result', 'improved', 'achieved', 'reduced', 'increased',
                'solved', 'fixed', 'deployed', 'implemented', 'completed',
                'successful', 'outcome', 'impact', 'metric', 'percent'
            ], words, ideal_per_100=1.0)

            # Also check if question type signals problem-solving expectation
            is_design = bool(re.search(r'\b(design|architect|build|implement|solve)\b', question, re.IGNORECASE))
            design_bonus = 10 if is_design else 0

            raw_score = (reasoning * 0.30 + tradeoffs * 0.25 + systematic * 0.25 + outcome * 0.20) * 100
            score     = int(min(100, raw_score + design_bonus))

            return StageResult("problem_solving", score, {
                "reasoning_depth":   round(reasoning, 3),
                "tradeoff_awareness":round(tradeoffs, 3),
                "systematic":        round(systematic, 3),
                "outcome_evidence":  round(outcome, 3),
                "is_design_question":is_design,
                "word_count":        words,
            })

        result = await loop.run_in_executor(None, _compute)
        result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE6] problem_solving={result.score} in {result.elapsed_ms}ms")
        return result

    @staticmethod
    def _pattern_density(text: str, patterns: List[str], word_count: int, ideal_per_100: float) -> float:
        hits = sum(1 for p in patterns if p in text)
        if word_count == 0:
            return 0.0
        density = hits / (word_count / 100)
        # Sigmoid-like normalisation: density/ideal → 0–1
        ratio = density / max(ideal_per_100, 0.01)
        return min(1.0, ratio ** 0.6)   # concave — first hits matter most


# ─── Stage 7 — STAR Evaluation ──────────────────────────────────────────────

class STAREvaluationStage:
    """
    Stage 7: Evaluate behavioral answers against the STAR framework.
    Only runs when is_behavioral=True (question is in behavioral stage).

    Checks for each STAR component via embedding similarity to canonical
    STAR descriptions, then validates with a lightweight LLM call.
    """

    STAR_SENTENCES = {
        "Situation": "setting the context, background, or situation of the story",
        "Task":      "describing the task, responsibility, or challenge assigned",
        "Action":    "explaining specific actions taken, steps followed, personal contribution",
        "Result":    "describing the outcome, result, impact, or what was achieved",
    }
    STAR_THRESHOLD = 0.28

    async def evaluate(
        self,
        question: str,
        answer_clean: str,
        loop: asyncio.AbstractEventLoop,
    ) -> StageResult:
        t0 = time.perf_counter()

        def _compute():
            model    = _get_embedding_model()
            ans_emb  = model.encode([answer_clean], normalize_embeddings=True)[0]
            star_embs= model.encode(list(self.STAR_SENTENCES.values()), normalize_embeddings=True)
            sims     = np.dot(star_embs, ans_emb)

            components = {}
            for (component, _), sim in zip(self.STAR_SENTENCES.items(), sims):
                components[component] = {
                    "present":    bool(sim >= self.STAR_THRESHOLD),
                    "similarity": round(float(sim), 3),
                }

            present_count = sum(1 for v in components.values() if v["present"])
            base_score    = int((present_count / 4) * 100)

            # Bonus: answer length (behavioral answers should be substantive)
            word_count   = len(answer_clean.split())
            length_bonus = min(10, max(0, (word_count - 50) / 10))
            score        = int(min(100, base_score + length_bonus))

            missing = [k for k, v in components.items() if not v["present"]]

            return StageResult("star_evaluation", score, {
                "components":      components,
                "present_count":   present_count,
                "missing":         missing,
                "word_count":      word_count,
            })

        result = await loop.run_in_executor(None, _compute)
        result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE7] star_evaluation={result.score} in {result.elapsed_ms}ms")
        return result


# ─── Stage 8 — Score Aggregation ────────────────────────────────────────────

class ScoreAggregator:
    """
    Stage 8: Weighted aggregation of all stage scores into final outputs.

    Weights differ by question type:
      Technical questions:  concept_coverage(25) technical_accuracy(35)
                            completeness(15) communication(15) problem_solving(10)
      Behavioral questions: concept_coverage(15) technical_accuracy(10)
                            completeness(20) communication(20) problem_solving(15)
                            star_evaluation(20)
      General questions:    concept_coverage(20) technical_accuracy(20)
                            completeness(25) communication(20) problem_solving(15)

    skill_score = technical_accuracy × 0.6 + concept_coverage × 0.4
    """

    TECHNICAL_WEIGHTS = {
        "concept_coverage":   0.25,
        "technical_accuracy": 0.35,
        "completeness":       0.15,
        "communication":      0.15,
        "problem_solving":    0.10,
    }
    BEHAVIORAL_WEIGHTS = {
        "concept_coverage":   0.15,
        "technical_accuracy": 0.10,
        "completeness":       0.20,
        "communication":      0.20,
        "problem_solving":    0.15,
        "star_evaluation":    0.20,
    }
    GENERAL_WEIGHTS = {
        "concept_coverage":   0.20,
        "technical_accuracy": 0.20,
        "completeness":       0.25,
        "communication":      0.20,
        "problem_solving":    0.15,
    }

    def aggregate(
        self,
        concept: StageResult,
        technical: StageResult,
        completeness: StageResult,
        communication: StageResult,
        problem_solving: StageResult,
        star: Optional[StageResult],
        is_behavioral: bool,
        is_technical: bool,
    ) -> Tuple[int, int, List[str], List[str]]:
        """
        Returns (question_score, skill_score, strengths, weaknesses).
        """
        scores = {
            "concept_coverage":   concept.score,
            "technical_accuracy": technical.score,
            "completeness":       completeness.score,
            "communication":      communication.score,
            "problem_solving":    problem_solving.score,
        }

        if is_behavioral and star:
            weights = self.BEHAVIORAL_WEIGHTS
            scores["star_evaluation"] = star.score
        elif is_technical:
            weights = self.TECHNICAL_WEIGHTS
        else:
            weights = self.GENERAL_WEIGHTS

        # Weighted sum (weights already sum to 1.0)
        question_score = int(sum(scores[k] * weights.get(k, 0) for k in scores))
        question_score = max(0, min(100, question_score))

        # Skill score: domain-specific technical competence
        skill_score = int(
            technical.score * 0.60 + concept.score * 0.40
        )
        skill_score = max(0, min(100, skill_score))

        strengths, weaknesses = self._generate_feedback(
            scores, is_behavioral, is_technical, concept, technical,
            completeness, communication, problem_solving, star
        )

        return question_score, skill_score, strengths, weaknesses

    @staticmethod
    def _generate_feedback(
        scores: Dict[str, float],
        is_behavioral: bool,
        is_technical: bool,
        concept: StageResult,
        technical: StageResult,
        completeness: StageResult,
        communication: StageResult,
        problem_solving: StageResult,
        star: Optional[StageResult],
    ) -> Tuple[List[str], List[str]]:
        strengths, weaknesses = [], []

        # Concept Coverage
        if scores["concept_coverage"] >= 70:
            covered = concept.details.get("concepts_covered", 0)
            total   = concept.details.get("concepts_checked", 1)
            strengths.append(f"Covered {covered}/{total} key concepts from the question and JD")
        elif scores["concept_coverage"] < 45:
            gaps = concept.details.get("concepts_checked", 0) - concept.details.get("concepts_covered", 0)
            weaknesses.append(f"Missed approximately {gaps} expected concepts — broaden your answer")

        # Technical Accuracy
        if scores["technical_accuracy"] >= 75:
            acc = technical.details.get("accurate_claims", [])
            if acc:
                strengths.append(f"Technically accurate: {acc[0][:80]}")
            else:
                strengths.append("Strong technical accuracy across the answer")
        elif scores["technical_accuracy"] < 50:
            errs = technical.details.get("inaccurate_claims", [])
            if errs:
                weaknesses.append(f"Technical inaccuracy: {errs[0][:80]}")
            else:
                weaknesses.append("Technical claims lacked precision or contained errors")

        # Completeness
        if scores["completeness"] >= 75:
            strengths.append("Addressed all parts of the question thoroughly")
        elif scores["completeness"] < 50:
            gaps = completeness.details.get("gaps", [])
            if gaps:
                weaknesses.append(f"Did not address: '{gaps[0][:60]}'")
            else:
                weaknesses.append("Answer was incomplete — some parts of the question were skipped")

        # Communication
        if scores["communication"] >= 75:
            strengths.append(f"Clear, well-structured communication (clarity={communication.details.get('clarity',0):.0f})")
        elif scores["communication"] < 50:
            if communication.details.get("structure", 0) < 50:
                weaknesses.append("Answer lacked logical structure — use signposting words")
            else:
                weaknesses.append("Communication could be clearer — avoid run-on sentences")

        # Problem Solving
        if scores["problem_solving"] >= 70:
            if problem_solving.details.get("tradeoff_awareness", 0) > 0.5:
                strengths.append("Demonstrated trade-off awareness and systematic reasoning")
            else:
                strengths.append("Showed structured problem-solving approach")
        elif scores["problem_solving"] < 40:
            if problem_solving.details.get("outcome_evidence", 0) < 0.3:
                weaknesses.append("Lacked outcome evidence — quantify results where possible")
            else:
                weaknesses.append("Problem-solving reasoning was surface-level — go deeper into your approach")

        # STAR (behavioral)
        if is_behavioral and star:
            missing = star.details.get("missing", [])
            if not missing:
                strengths.append("Complete STAR response — Situation, Task, Action, and Result all present")
            elif len(missing) <= 1:
                strengths.append(f"Strong STAR answer; only '{missing[0]}' could be expanded")
            else:
                weaknesses.append(f"STAR response missing: {', '.join(missing)} — structure your behavioral answers")

        return strengths[:4], weaknesses[:4]


# ─── Stage 9 — Confidence Score ─────────────────────────────────────────────

class ConfidenceScoreStage:
    """
    Stage 9: Estimate how confident the evaluation result is.

    Factors:
      • Answer length (longer = more signal)
      • Stage score spread (high variance = uncertain evaluation)
      • Concept coverage ratio (low coverage = uncertain relevance)
      • Technical accuracy parse success
      • Whether answer is on-topic (cosine sim to question > 0.25)
    """

    async def evaluate(
        self,
        answer_clean: str,
        question: str,
        stage_scores: List[float],
        concept_coverage_ratio: float,
        technical_parse_success: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> float:
        """Returns a confidence float in [0, 1]."""
        t0 = time.perf_counter()

        def _compute():
            word_count = len(answer_clean.split())

            # Factor 1: answer length (0–1)
            length_conf = min(1.0, word_count / 80)

            # Factor 2: stage score spread (low spread → high confidence)
            if len(stage_scores) >= 2:
                spread    = max(stage_scores) - min(stage_scores)
                spread_conf = max(0.3, 1.0 - spread / 100)
            else:
                spread_conf = 0.7

            # Factor 3: concept coverage
            coverage_conf = 0.5 + concept_coverage_ratio * 0.5

            # Factor 4: LLM parse success
            parse_conf = 1.0 if technical_parse_success else 0.6

            # Factor 5: on-topic check (embedding cosine)
            model    = _get_embedding_model()
            embs     = model.encode([question, answer_clean], normalize_embeddings=True)
            on_topic = float(np.dot(embs[0], embs[1]))
            topic_conf = min(1.0, max(0.1, on_topic / 0.4))  # 0.4 = decent on-topic

            confidence = (
                length_conf   * 0.25 +
                spread_conf   * 0.20 +
                coverage_conf * 0.25 +
                parse_conf    * 0.15 +
                topic_conf    * 0.15
            )
            return round(min(1.0, max(0.0, confidence)), 3)

        confidence = await loop.run_in_executor(None, _compute)
        elapsed    = int((time.perf_counter() - t0) * 1000)
        logger.debug(f"[STAGE9] confidence={confidence:.3f} in {elapsed}ms")
        return confidence


# ─── Orchestrator ────────────────────────────────────────────────────────────

class SemanticEvaluationEngine:
    """
    Orchestrates the full 9-stage pipeline for a single Q&A pair.

    Usage (from a FastAPI BackgroundTask):
        engine = SemanticEvaluationEngine(llm_fn=llm_service.generate_short)
        result = await engine.evaluate_question(
            question_id=3,
            interview_id="abc123",
            question="Explain how FAISS works...",
            transcript_raw=raw_stt_text,
            job_description=jd,
            resume_context=resume_text,
            stage="technical",
        )
    """

    def __init__(self, llm_fn):
        """
        llm_fn: a sync callable (prompt: str) -> str
                (e.g. OllamaService().generate_short)
        """
        self.llm_fn          = llm_fn
        self.cleaner         = TranscriptCleaner()
        self.concept_stage   = ConceptCoverageStage()
        self.technical_stage = TechnicalAccuracyStage()
        self.completeness_st = CompletenessStage()
        self.communication_st= CommunicationStage()
        self.problem_solving = ProblemSolvingStage()
        self.star_stage      = STAREvaluationStage()
        self.aggregator      = ScoreAggregator()
        self.confidence_st   = ConfidenceScoreStage()

    async def evaluate_question(
        self,
        question_id: int,
        interview_id: str,
        question: str,
        transcript_raw: str,
        job_description: str,
        resume_context: str,
        stage: str,                     # "technical", "behavioral", etc.
    ) -> QuestionEvalResult:
        """
        Full 9-stage evaluation. Returns QuestionEvalResult.
        Designed to be called as a BackgroundTask — does not block HTTP response.
        """
        t_total = time.perf_counter()
        loop    = asyncio.get_event_loop()

        is_behavioral = stage == "behavioral"
        is_technical  = stage in ("technical", "resume")

        # ── Stage 1: Transcript Cleaning (sequential — input to all others) ──
        transcript_clean = await loop.run_in_executor(
            None, TranscriptCleaner.clean, transcript_raw
        )
        logger.info(f"[SEM-EVAL] [{interview_id}] Q{question_id} stage='{stage}' "
                    f"raw={len(transcript_raw)}ch clean={len(transcript_clean)}ch")

        # ── Stages 2–6: Parallel ────────────────────────────────────────────
        parallel_tasks = asyncio.gather(
            self.concept_stage.evaluate(question, transcript_clean, job_description, resume_context, loop),
            self.technical_stage.evaluate(question, transcript_clean, job_description, self.llm_fn, loop),
            self.completeness_st.evaluate(question, transcript_clean, loop),
            self.communication_st.evaluate(transcript_raw, transcript_clean, loop),
            self.problem_solving.evaluate(question, transcript_clean, loop),
            return_exceptions=True,
        )
        results = await parallel_tasks

        # Unpack with fallbacks for any stage that raised
        concept_res    = results[0] if not isinstance(results[0], Exception) else StageResult("concept_coverage",   50)
        technical_res  = results[1] if not isinstance(results[1], Exception) else StageResult("technical_accuracy", 50)
        completeness_r = results[2] if not isinstance(results[2], Exception) else StageResult("completeness",       50)
        communication_r= results[3] if not isinstance(results[3], Exception) else StageResult("communication",      50)
        problemsolving_r=results[4] if not isinstance(results[4], Exception) else StageResult("problem_solving",    50)

        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"[SEM-EVAL] Stage {i+2} failed: {r}")

        # ── Stage 7: STAR (behavioral only) ─────────────────────────────────
        star_res = None
        if is_behavioral:
            star_res = await self.star_stage.evaluate(question, transcript_clean, loop)

        # ── Stage 8: Score Aggregation ───────────────────────────────────────
        question_score, skill_score, strengths, weaknesses = self.aggregator.aggregate(
            concept_res, technical_res, completeness_r,
            communication_r, problemsolving_r, star_res,
            is_behavioral, is_technical,
        )

        # ── Stage 9: Confidence Score ─────────────────────────────────────────
        stage_scores = [
            concept_res.score, technical_res.score,
            completeness_r.score, communication_r.score, problemsolving_r.score,
        ]
        tech_parse_ok = "note" not in technical_res.details

        confidence = await self.confidence_st.evaluate(
            transcript_clean, question, stage_scores,
            concept_res.details.get("coverage_ratio", 0.5),
            tech_parse_ok, loop,
        )

        total_ms = int((time.perf_counter() - t_total) * 1000)
        logger.info(
            f"[SEM-EVAL] [{interview_id}] Q{question_id} DONE in {total_ms}ms — "
            f"q_score={question_score} skill={skill_score} conf={confidence:.2f}"
        )

        return QuestionEvalResult(
            question_id       = question_id,
            interview_id      = interview_id,
            question          = question,
            transcript_raw    = transcript_raw,
            transcript_clean  = transcript_clean,
            concept_coverage  = concept_res,
            technical_accuracy= technical_res,
            completeness      = completeness_r,
            communication     = communication_r,
            problem_solving   = problemsolving_r,
            star_evaluation   = star_res,
            question_score    = question_score,
            skill_score       = skill_score,
            confidence_score  = confidence,
            strengths         = strengths,
            weaknesses        = weaknesses,
            total_elapsed_ms  = total_ms,
            is_behavioral     = is_behavioral,
        )
