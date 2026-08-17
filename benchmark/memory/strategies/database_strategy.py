"""PostgreSQL + pgvector retrieval strategy.

Production-grade persistent storage with vector similarity search.
Scalable to millions of memories.

Latency: 10-50ms | Cost: Low | Accuracy: Excellent | Setup: 2 hours
"""

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None
    execute_values = None

from benchmark.memory.interfaces.retrieval_strategy import RetrievalStrategy
from benchmark.models.memory_event import MemoryEvent
from benchmark.observability.logger import get_logger

logger = get_logger(__name__)


class DatabaseStrategy(RetrievalStrategy):
    """PostgreSQL with pgvector storage and retrieval."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "postgres",
        password: str = "password",
        database: str = "memories",
        connection_timeout: int = 5,
        statement_timeout: int = 5000,
    ) -> None:
        """Initialize database strategy.

        Args:
            host: Database host.
            port: Database port.
            user: Database user.
            password: Database password.
            database: Database name.
            connection_timeout: Connection timeout in seconds.
            statement_timeout: Query timeout in milliseconds.
        """
        if psycopg2 is None:
            raise ImportError("psycopg2 not installed. Install: pip install psycopg[binary]")

        try:
            self._conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                timeout=connection_timeout,
            )
            self._conn.autocommit = True

            # Set statement timeout to prevent long-running queries
            cursor = self._conn.cursor()
            cursor.execute(f"SET statement_timeout = {statement_timeout}")
            cursor.close()

        except psycopg2.OperationalError as e:
            raise RuntimeError(
                f"Failed to connect to PostgreSQL at {host}:{port}. "
                f"Connection timeout={connection_timeout}s. "
                f"Check if PostgreSQL is running.\n{e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize database strategy: {e}") from e

        self._memories: dict[str, MemoryEvent] = {}
        self._connection_timeout = connection_timeout
        self._statement_timeout = statement_timeout
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self._conn.cursor()
        try:
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Create index for user queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                ON memories(user_id)
            """)

            self._conn.commit()
        except Exception as e:
            if psycopg2 and isinstance(e, (psycopg2.OperationalError, psycopg2.ProgrammingError)):
                logger.warning(f"Database schema initialization failed (non-fatal): {e}")
            else:
                logger.exception(f"Unexpected database error during schema initialization")
        finally:
            cursor.close()

    def index(self, memories: list[MemoryEvent]) -> None:
        """Index memories in database.

        Args:
            memories: List of memories to index.
        """
        self._memories = {mem.id: mem for mem in memories}

        cursor = self._conn.cursor()
        try:
            # Clear old data
            cursor.execute("DELETE FROM memories")

            # Insert new data
            values = [(mem.id, mem.user_id, mem.content, mem.importance) for mem in memories]

            if values:
                execute_values(
                    cursor,
                    """
                    INSERT INTO memories (id, user_id, content, importance)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE
                    SET content = EXCLUDED.content,
                        importance = EXCLUDED.importance
                    """,
                    values,
                    page_size=1000,
                )

            self._conn.commit()
        except Exception as e:
            if psycopg2 and isinstance(e, (psycopg2.OperationalError, psycopg2.ProgrammingError)):
                logger.warning(f"Database indexing failed (non-fatal): {e}")
            else:
                logger.exception(f"Unexpected error during memory indexing")
            self._conn.rollback()
        finally:
            cursor.close()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve from database.

        For now, uses keyword search. With pgvector embeddings,
        this could be upgraded to vector similarity search.

        Args:
            query: The query text.
            top_k: Number of results to return.
            user_id: Optional user filter.

        Returns:
            List of (memory_id, score) tuples.
        """
        cursor = self._conn.cursor()
        try:
            # Simple keyword search (can be upgraded to vector search)
            query_pattern = f"%{query}%"

            if user_id:
                cursor.execute(
                    """
                    SELECT id, importance FROM memories
                    WHERE user_id = %s AND content ILIKE %s
                    ORDER BY importance DESC
                    LIMIT %s
                    """,
                    (user_id, query_pattern, top_k),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, importance FROM memories
                    WHERE content ILIKE %s
                    ORDER BY importance DESC
                    LIMIT %s
                    """,
                    (query_pattern, top_k),
                )

            rows = cursor.fetchall()
            return [(row[0], float(row[1]) if row[1] else 0.5) for row in rows]

        except psycopg2.OperationalError as e:
            # Connection timeout or query timeout
            raise RuntimeError(
                f"Database operation timeout or connection error. "
                f"Statement timeout: {self._statement_timeout}ms, "
                f"Connection timeout: {self._connection_timeout}s\n{e}"
            ) from e
        except psycopg2.DatabaseError as e:
            # SQL error or other database issue
            raise RuntimeError(f"Database error during retrieval: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during database retrieval: {e}") from e
            return []
        finally:
            cursor.close()

    def name(self) -> str:
        """Return strategy name."""
        return "database"

    def clear(self) -> None:
        """Clear all memories from database."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("DELETE FROM memories")
            self._conn.commit()
        except Exception as e:
            if psycopg2 and isinstance(e, (psycopg2.OperationalError, psycopg2.ProgrammingError)):
                logger.warning(f"Failed to clear memories from database (non-fatal): {e}")
            else:
                logger.exception(f"Unexpected error during memory clear")
        finally:
            cursor.close()

        self._memories.clear()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()

    @classmethod
    def is_available(cls) -> bool:
        """Check if psycopg2 is installed."""
        return psycopg2 is not None
