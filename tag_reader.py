import sqlite3
import numpy as np

conn = sqlite3.connect("images.db")
conn.row_factory = sqlite3.Row

cur = conn.execute(
    "SELECT path, tags, caption FROM images LIMIT 10"
)

for row in cur:
    print(row["path"], row["tags"])

def search_by_tag(conn, tag):
    return conn.execute(
        "SELECT path FROM images WHERE tags LIKE ?",
        (f"%{tag}%",)
    ).fetchall()

rows = search_by_tag(conn, "1girl")



row = conn.execute(
    "SELECT embedding FROM images WHERE id = 1"
).fetchone()

vec = np.frombuffer(row["embedding"], dtype=np.float32)
print(vec.shape)
