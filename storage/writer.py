import numpy as np

def write_image(conn, meta, embedding):
    conn.execute(
        """
        INSERT OR IGNORE INTO images
        (path, md5, width, height, format, tags, caption, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta["path"],
            meta["hash"],
            meta["w"],
            meta["h"],
            meta["format"],
            meta["tags"],
            meta["caption"],
            embedding.tobytes()
        )
    )
    conn.commit()