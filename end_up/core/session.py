# session.py - 会话管理模块
import sqlite3
import uuid
from datetime import datetime

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id TEXT PRIMARY KEY, title TEXT, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id TEXT PRIMARY KEY, session_id TEXT, role TEXT,
                  content TEXT, created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

def create_session(title="新对话"):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO sessions VALUES (?, ?, ?)',
                 (session_id, title, datetime.now()))
    conn.commit()
    conn.close()
    return session_id

def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute('''
        SELECT DISTINCT s.id, s.title, s.created_at
        FROM sessions s
        INNER JOIN messages m ON s.id = m.session_id
        ORDER BY s.created_at DESC
    ''').fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('INSERT INTO messages VALUES (?, ?, ?, ?, ?)',
                 (str(uuid.uuid4()), session_id, role, content, datetime.now()))
    conn.commit()
    conn.close()

def get_messages(session_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        'SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC',
        (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]

def update_session_title(session_id, title):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE sessions SET title = ? WHERE id = ?', (title, session_id))
    conn.commit()
    conn.close()

def delete_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()
