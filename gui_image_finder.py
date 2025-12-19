# gui_image_finder.py
# 一个最小但工程化的图片查找 GUI（SQLite + tags + caption）
# 技术选型：Tkinter（内置、稳定、适合工具型应用）

import sqlite3
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pathlib import Path

DB_PATH = "images.db"      # 你的 sqlite db
IMAGE_MAX_SIZE = (512, 512)

# -----------------------------
# 数据层
# -----------------------------
class ImageDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def search(self, tag=None, text=None, limit=200):
        sql = "SELECT path, tags, caption FROM images WHERE 1=1"
        params = []

        if tag:
            sql += " AND tags LIKE ?"
            params.append(f"%{tag}%")

        if text:
            sql += " AND caption LIKE ?"
            params.append(f"%{text}%")

        sql += " LIMIT ?"
        params.append(limit)

        return self.conn.execute(sql, params).fetchall()


# -----------------------------
# GUI
# -----------------------------
class ImageFinderApp(tk.Tk):
    def __init__(self, db: ImageDB):
        super().__init__()
        self.db = db
        self.title("Image Finder")
        self.geometry("1200x800")

        self._build_ui()
        self.images_cache = []  # 防止 PhotoImage 被 GC

    def _build_ui(self):
        # ===== 查询栏 =====
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(top, text="Tag:").pack(side=tk.LEFT)
        self.tag_entry = ttk.Entry(top, width=20)
        self.tag_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(top, text="Caption:").pack(side=tk.LEFT)
        self.text_entry = ttk.Entry(top, width=30)
        self.text_entry.pack(side=tk.LEFT, padx=5)

        ttk.Button(top, text="Search", command=self.search).pack(side=tk.LEFT, padx=10)

        # ===== 结果区 =====
        self.canvas = tk.Canvas(self)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # -----------------------------
    # 查询 & 显示
    # -----------------------------
    def search(self):
        tag = self.tag_entry.get().strip()
        text = self.text_entry.get().strip()

        rows = self.db.search(tag=tag or None, text=text or None)

        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        self.images_cache.clear()

        for i, row in enumerate(rows):
            self._add_result(row, i)

    def _add_result(self, row, idx):
        frame = ttk.Frame(self.scrollable_frame, padding=5)
        frame.grid(row=idx // 3, column=idx % 3, sticky="n")

        path = Path(row["path"])
        if not path.exists():
            return

        img = Image.open(path).convert("RGB")
        img.thumbnail(IMAGE_MAX_SIZE)
        tk_img = ImageTk.PhotoImage(img)
        self.images_cache.append(tk_img)

        lbl = ttk.Label(frame, image=tk_img)
        lbl.pack()

        ttk.Label(frame, text=path.name, wraplength=200).pack()
        ttk.Label(frame, text=row["tags"], wraplength=200, foreground="gray").pack()


# -----------------------------
# 启动
# -----------------------------
if __name__ == "__main__":
    db = ImageDB(DB_PATH)
    app = ImageFinderApp(db)
    app.mainloop()
