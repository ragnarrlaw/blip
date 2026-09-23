from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from config.conf import Config, get_config

Base = declarative_base()

class DatabaseManager:
    def __init__(self, cfg: Config):
        self.config = cfg
        connect_args = {}

        # FastAPI's concurrent requests require check_same_thread=False for SQLite
        if self.config.blip_database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine = create_engine(
            self.config.blip_database_url,
            connect_args=connect_args,
            echo=self.config.blip_sql_echo
        )

        # Attach the PRAGMA event listener only to THIS specific engine instance,
        # ensuring it doesn't bleed into other engines during testing.
        if self.config.blip_database_url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._set_sqlite_pragma)

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def _set_sqlite_pragma(self, dbapi_connection, connection_record):
        """Enforce SQLite concurrency settings based on the config object."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute(f"PRAGMA synchronous={self.config.blip_sqlite_synchronous}")
        cursor.execute(f"PRAGMA busy_timeout={self.config.blip_sqlite_busy_timeout_ms}")
        cursor.close()

    def get_session(self):
        """FastAPI dependency for yielding database sessions."""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()

    def create_tables(self):
        """Build all tables defined in models.py."""
        # Note: Ensure models are imported before calling this
        Base.metadata.create_all(bind=self.engine)


# Expose a single instance for the main application lifecycle
db_manager = DatabaseManager(get_config())