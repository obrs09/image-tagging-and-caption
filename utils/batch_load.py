from itertools import islice
from tkinter import Image
# -----------------------------
# 批量工具
# -----------------------------
def batch(iterable, n=16):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


# -----------------------------
# 图片加载
# -----------------------------
def load_images_batches(paths, size=(448,448)):
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail(size)
            imgs.append(img)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            imgs.append(None)
    return imgs