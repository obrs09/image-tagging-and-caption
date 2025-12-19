import sqlite3
from config import DB_PATH

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY,
        path TEXT UNIQUE,
        hash TEXT,
        width INTEGER,
        height INTEGER,
        format TEXT,
        tags TEXT,
        caption TEXT,
        embedding BLOB
    )
    """)
    return conn