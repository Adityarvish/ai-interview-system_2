import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / '.env')


class Config:
    # MySQL
    MYSQL_HOST     = os.environ.get('MYSQL_HOST',     'localhost')
    MYSQL_PORT     = int(os.environ.get('MYSQL_PORT', 3306))
    MYSQL_USER     = os.environ.get('MYSQL_USER',     'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DB       = os.environ.get('MYSQL_DB',       'ai_interview_db')
    MYSQL_URL      = os.environ.get(
        'MYSQL_URL',
        f"mysql+pymysql://{os.environ.get('MYSQL_USER', 'root')}:"
        f"{os.environ.get('MYSQL_PASSWORD', '')}@"
        f"{os.environ.get('MYSQL_HOST', 'localhost')}:"
        f"{os.environ.get('MYSQL_PORT', '3306')}/"
        f"{os.environ.get('MYSQL_DB', 'ai_interview_db')}?charset=utf8mb4"
    )

    # ── Groq Cloud API ────────────────────────────────────────────────────────
    GROQ_API_KEY        = os.environ.get('GROQ_API_KEY', '')
    GROQ_PRIMARY_MODEL  = os.environ.get('GROQ_PRIMARY_MODEL',  'llama-3.3-70b-versatile')
    GROQ_FALLBACK_MODEL = os.environ.get('GROQ_FALLBACK_MODEL', 'llama-3.1-8b-instant')

    # FIX #8: renamed from FLASK_HOST / FLASK_PORT — this is a FastAPI/uvicorn app.
    HOST = os.environ.get('HOST', os.environ.get('FLASK_HOST', '0.0.0.0'))
    PORT = int(os.environ.get('PORT', os.environ.get('FLASK_PORT', 5000)))

    # Interview settings
    MAX_INTERVIEW_DURATION = int(os.environ.get('MAX_INTERVIEW_DURATION', 2700))  # 45 minutes

    # File paths
    UPLOAD_FOLDER  = ROOT_DIR / 'uploads'
    RESUME_FOLDER  = UPLOAD_FOLDER / 'resumes'
    AUDIO_FOLDER   = UPLOAD_FOLDER / 'audio'

    # Allowed extensions
    ALLOWED_RESUME_EXTENSIONS = {'pdf', 'txt'}
    ALLOWED_AUDIO_EXTENSIONS  = {'webm', 'wav', 'ogg', 'mp3'}

    # Max file sizes (in bytes)
    MAX_RESUME_SIZE = 10 * 1024 * 1024   # 10 MB
    MAX_AUDIO_SIZE  = 50 * 1024 * 1024   # 50 MB


# Create upload directories
Config.UPLOAD_FOLDER.mkdir(exist_ok=True)
Config.RESUME_FOLDER.mkdir(exist_ok=True)
Config.AUDIO_FOLDER.mkdir(exist_ok=True)