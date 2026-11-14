"""Database connectivity helpers using SQLAlchemy 2.0.

Provides convenience functions to create an :class:`~sqlalchemy.engine.Engine`
and to obtain context-managed :class:`~sqlalchemy.orm.Session` instances.

Usage::

    from src.common.db import get_engine, get_session

    engine = get_engine()          # reads URL from project config
    with get_session() as session:
        result = session.execute(text("SELECT 1"))
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session, sessionmaker

from src.common.config_loader import load_config
from src.common.exceptions import ConfigError
from src.common.logger import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def get_engine(url: str | None = None) -> Engine:
    """Create or return a cached SQLAlchemy :class:`Engine`.

    When *url* is ``None`` the database URL is read from the project
    configuration file under the key ``database.url``.

    Args:
        url: Optional SQLAlchemy connection URL.  If omitted the URL is
            resolved via :func:`src.common.config_loader.load_config`.

    Returns:
        A :class:`sqlalchemy.engine.Engine` instance ready for use.

    Raises:
        ConfigError: If no URL is provided and the config file does not
            contain a ``database.url`` entry.
    """
    global _engine, _SessionFactory  # noqa: PLW0603

    if _engine is not None:
        return _engine

    if url is None:
        config: dict[str, Any] = load_config()
        db_config: dict[str, Any] | None = config.get("database")
        if db_config is None or "url" not in db_config:
            raise ConfigError(
                "Database URL not found in configuration. "
                "Set 'database.url' in config/settings.yaml or pass a URL explicitly."
            )
        url = db_config["url"]

    logger.info("Creating database engine.")
    _engine = create_engine(url, pool_pre_ping=True)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional :class:`Session` that auto-closes on exit.

    The session is committed when the ``with`` block exits normally and
    rolled back if an exception propagates.

    Yields:
        A :class:`sqlalchemy.orm.Session` bound to the project engine.

    Raises:
        ConfigError: Propagated from :func:`get_engine` if the database URL
            cannot be resolved.
    """
    engine = get_engine()

    global _SessionFactory  # noqa: PLW0603
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    session: Session = _SessionFactory()
    logger.debug("Database session opened.")

    try:
        yield session
        session.commit()
        logger.debug("Database session committed.")
    except Exception:
        session.rollback()
        logger.exception("Database session rolled back due to an error.")
        raise
    finally:
        session.close()
        logger.debug("Database session closed.")
