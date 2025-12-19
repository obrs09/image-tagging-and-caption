from PIL import Image, UnidentifiedImageError
import numpy as np
import os

def load_image(path):
    img = Image.open(path).convert("RGB")
    try:
        # 尝试打开并转换
        with Image.open(path) as img:
            img.convert("RGB")
        img = Image.open(path).convert("RGB")
        
    except (UnidentifiedImageError, OSError, ValueError) as e:
        # 捕获特定的错误：不是图片、文件损坏、或转换错误
        print(f"跳过无效文件: {path} | 原因: {e}")
        # continue

    except Exception as e:
        # 捕获其他意料之外的错误
        print(f"发生未知错误: {path} | {e}")
        # continue
    return img