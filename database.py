import json
import hashlib
import os
import sys
from pathlib import Path
import psycopg2
from contextlib import contextmanager
from langchain_core.messages import HumanMessage, AIMessage
from psycopg2.pool import ThreadedConnectionPool

def get_streamlit_secret(name, default=None):
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def is_running_in_container():
    return os.path.exists("/.dockerenv")


def normalize_postgres_uri(uri):
    if not uri or is_running_in_container():
        return uri
    return uri.replace("@postgres_shared:5432/", "@localhost:5433/")


env_path = Path(__file__).resolve().parent / ".env"
sys.path.insert(0, str(env_path))

try:
    from key import LOCAL_DATABASE_URL as KEY_FILE_LOCAL_DATABASE_URL
except ImportError:
    KEY_FILE_LOCAL_DATABASE_URL = None

PG_URI = normalize_postgres_uri(
    os.getenv("LOCAL_DATABASE_URL")
    or get_streamlit_secret("LOCAL_DATABASE_URL")
    or os.getenv("POSTGRES_LOCAL_URL")
    or get_streamlit_secret("POSTGRES_LOCAL_URL")
    or KEY_FILE_LOCAL_DATABASE_URL
    or "postgresql://admin:admin@postgres_shared:5432/rag_lakehouse"
)
DB_POOL_MIN_CONN = int(os.getenv("DB_POOL_MIN_CONN", "1"))
DB_POOL_MAX_CONN = int(os.getenv("DB_POOL_MAX_CONN", "5"))
_db_pool = None


def get_db_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = ThreadedConnectionPool(
            DB_POOL_MIN_CONN,
            DB_POOL_MAX_CONN,
            dsn=PG_URI,
        )
    return _db_pool


@contextmanager
def get_db_connection():
    pool = get_db_pool()
    conn = pool.getconn()
    close_connection = False
    try:
        yield conn
        conn.commit()
    except Exception:
        close_connection = True
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pool.putconn(conn, close=close_connection or bool(conn.closed))

def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    session_key TEXT UNIQUE,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                ALTER TABLE chat_sessions
                ADD COLUMN IF NOT EXISTS session_key TEXT
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id
                ON chat_sessions (user_id)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_session_counters (
                    user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    last_session_number INTEGER NOT NULL DEFAULT 0
                )
            """)
            cur.execute("""
                WITH ranked AS (
                    SELECT cs.id,
                           u.username,
                           ROW_NUMBER() OVER (
                               PARTITION BY cs.user_id
                               ORDER BY cs.created_at, cs.id
                           ) AS seq
                    FROM chat_sessions cs
                    JOIN users u ON u.id = cs.user_id
                    WHERE cs.session_key IS NULL
                )
                UPDATE chat_sessions cs
                SET session_key = ranked.username || '_' || ranked.seq
                FROM ranked
                WHERE cs.id = ranked.id
            """)
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_sessions_session_key
                ON chat_sessions (session_key)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    source_nodes JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                UPDATE chat_history ch
                SET session_id = cs.session_key
                FROM chat_sessions cs
                WHERE ch.session_id = cs.id::text
                  AND cs.session_key IS NOT NULL
            """)
            cur.execute("""
                INSERT INTO chat_session_counters (user_id, last_session_number)
                SELECT user_id, MAX(session_number)
                FROM (
                    SELECT cs.user_id,
                           substring(cs.session_key from length(u.username) + 2)::integer AS session_number
                    FROM chat_sessions cs
                    JOIN users u ON u.id = cs.user_id
                    WHERE left(cs.session_key, length(u.username) + 1) = u.username || '_'
                      AND substring(cs.session_key from length(u.username) + 2) ~ '^[0-9]+$'
                ) numbered_sessions
                GROUP BY user_id
                ON CONFLICT (user_id) DO UPDATE
                SET last_session_number = GREATEST(
                    chat_session_counters.last_session_number,
                    EXCLUDED.last_session_number
                )
            """)

def _hash_password(password: str, salt_hex: str | None = None):
    if salt_hex is None:
        salt = os.urandom(16)
        salt_hex = salt.hex()
    else:
        salt = bytes.fromhex(salt_hex)
    hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hash_bytes.hex(), salt_hex

def create_user(username: str, password: str):
    password_hash, password_salt = _hash_password(password)
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash, password_salt)
                VALUES (%s, %s, %s)
                RETURNING id
            """, (username, password_hash, password_salt))
            user_id = cur.fetchone()[0]

    return user_id

def verify_user(username: str, password: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, password_hash, password_salt
                FROM users
                WHERE username = %s
            """, (username,))
            row = cur.fetchone()
    if not row:
        return None
    user_id, stored_hash, stored_salt = row
    computed_hash, _ = _hash_password(password, stored_salt)
    if computed_hash == stored_hash:
        return user_id
    return None

def create_session(user_id: int, username: str, title: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_session_counters (user_id, last_session_number)
                VALUES (%s, 1)
                ON CONFLICT (user_id) DO UPDATE
                SET last_session_number = chat_session_counters.last_session_number + 1
                RETURNING last_session_number
            """, (user_id,))
            next_session_number = cur.fetchone()[0]
            session_key = f"{username}_{next_session_number}"
            cur.execute("""
                INSERT INTO chat_sessions (user_id, session_key, title)
                VALUES (%s, %s, %s)
                RETURNING session_key
            """, (user_id, session_key, title))
            created_key = cur.fetchone()[0]

    return created_key

def update_session_title(session_id: str, title: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE chat_sessions
                SET title = %s
                WHERE session_key = %s
            """, (title, session_id))


def get_session_title(session_id: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT title
                FROM chat_sessions
                WHERE session_key = %s
            """, (session_id,))
            row = cur.fetchone()
    return row[0] if row else None

def list_sessions(user_id: int):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(session_key, id::text), title, created_at
                FROM chat_sessions
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            rows = cur.fetchall()
    return [
        {"id": row[0], "title": row[1], "created_at": row[2]}
        for row in rows
    ]

def delete_session(user_id: int, session_id: str):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM chat_history
                WHERE session_id = %s
            """, (session_id,))
            cur.execute("""
                DELETE FROM chat_sessions
                WHERE session_key = %s AND user_id = %s
            """, (session_id, user_id))


def load_messages_from_history(session_id: str):
    messages = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_query, ai_response, source_nodes, created_at
                FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at ASC
            """, (session_id,))
            rows = cur.fetchall()
    for user_query, ai_response, source_nodes, created_at in rows:
        messages.append({
            "role": "user",
            "content": user_query,
            "source_nodes": None,
            "created_at": created_at,
        })
        messages.append({
            "role": "assistant",
            "content": ai_response,
            "source_nodes": source_nodes,
            "created_at": created_at,
        })
    return messages

def save_chat_turn_to_db(session_id: str, user_query: str, ai_response: str, source_nodes: list | None):
    """Lưu trọn bộ Hỏi - Đáp - Nguồn vào cùng 1 dòng"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history (session_id, user_query, ai_response, source_nodes) 
                VALUES (%s, %s, %s, %s)
            """, (
                session_id, 
                user_query, 
                ai_response, 
                json.dumps(source_nodes, ensure_ascii=False) if source_nodes else None
            ))


def load_history_from_db(session_id: str, limit: int = 5):
    """Tải lại lịch sử. limit=5 nghĩa là tải 5 cặp Hỏi-Đáp (Tương đương 10 tin nhắn)"""
    history = []
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT user_query, ai_response FROM (
                    SELECT user_query, ai_response, created_at 
                    FROM chat_history 
                    WHERE session_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                ) AS sub
                ORDER BY created_at ASC
            """, (session_id, limit))
            
            rows = cur.fetchall()
            for user_query, ai_response in rows:
                # Trích xuất 1 dòng ra thành 2 object gửi cho LangChain
                history.append(HumanMessage(content=user_query))
                history.append(AIMessage(content=ai_response))
    return history
