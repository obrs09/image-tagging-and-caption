import numpy as np

def write_image(conn, meta, embedding):
    conn.execute(
        """
        INSERT OR IGNORE INTO images
        (path, sha256, width, height, format, tags, caption, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta["path"],
            meta["sha256"],
            meta["w"],
            meta["h"],
            meta["format"],
            meta["tags"],
            meta["caption"],
            embedding.tobytes()
        )
    )
    conn.commit()