"""MySQL database connection using SQLAlchemy."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config.settings import Config
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

engine = create_engine(
    Config.MYSQL_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=10,
    max_overflow=20,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables if they don't exist."""
    from config import models  # noqa: F401 – ensure models are registered
    Base.metadata.create_all(bind=engine)
    logger.info("MySQL tables created / verified.")


def get_db():
    """Yield a SQLAlchemy session; close it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
