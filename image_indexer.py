import os
import sqlite3
import torch
import cv2  # 用于处理视频抽帧
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import onnxruntime as ort

# --- 配置 ---
DB_PATH = "media_library.db"
SOURCE_DIR = r"E:\e\img"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 

# 定义支持的格式
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv'}
ANIMATED_EXTS = {'.gif', '.webp'} # 某些webp是动图

def get_frame_from_video(video_path):
    """从视频中提取中间帧"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2) # 取中间一帧
    ret, frame = cap.read()
    cap.release()
    if ret:
        # 转换 BGR 到 RGB 并转为 PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    return None

def load_and_preprocess(file_path):
    """统一处理不同格式的输入"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext in IMAGE_EXTS:
            img = Image.open(file_path)
            # 处理动图 GIF/WebP，只取第一帧
            if hasattr(img, 'is_animated') and img.is_animated:
                img.seek(0)
            return img.convert("RGB")
        
        elif ext in VIDEO_EXTS:
            return get_frame_from_video(file_path)
            
        else:
            return None # 不支持的格式
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 数据库与推理逻辑 (保持与前文类似，增加格式字段) ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            type TEXT,         -- image, video, gif
            tags TEXT,
            caption TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

# --- 2. 加载模型 ---
# Florence-2
print("Loading Florence-2...")
f2_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True).to(DEVICE).eval()
f2_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

# CLIP (向量化)
print("Loading CLIP...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# WD1.4 Tagger (使用 ONNX)
# 请先下载模型文件: wd-v1-4-vit-tagger-v2.onnx 和 selected_tags.csv
print("Loading WD1.4 Tagger...")
wd_session = ort.InferenceSession("wd-v1-4-vit-tagger-v2.onnx", providers=['CUDAExecutionProvider'])
# 假设你已经有了标签列表 tags_list

def main():
    conn = init_db()
    
    # 1. 扫描文件并过滤
    all_files = []
    supported_exts = IMAGE_EXTS | VIDEO_EXTS
    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} valid files. Starting...")

    # 2. 批处理
    for i in tqdm(range(0, len(all_files), BATCH_SIZE)):
        paths = all_files[i:i+BATCH_SIZE]
        images = []
        valid_paths = []
        
        for p in paths:
            img = load_and_preprocess(p)
            if img:
                images.append(img)
                valid_paths.append(p)
        
        if not images: continue

        with torch.no_grad():
            # A. Florence-2 描述
            inputs = f2_processor(text="<DETAILED_CAPTION>", images=images, return_tensors="pt", padding=True).to(DEVICE)
            generated_ids = f2_model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024)
            captions = f2_processor.batch_decode(generated_ids, skip_special_tokens=True)

            # B. CLIP 向量化
            clip_inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
            image_embeds = clip_model.get_image_features(**clip_inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) # 归一化
            embeddings = image_embeds.cpu().numpy()

            # C. WD1.4 Tagger (简单演示逻辑，实际需Resize到448x448)
            # 这里仅占位，实际操作建议封装成单独的推理类
            tags_placeholder = ["tags_example"] * len(images)
        
        # --- 存入数据库 ---
        cursor = conn.cursor()
        for idx, path in enumerate(valid_paths):
            ext = os.path.splitext(path)[1].lower()
            media_type = "video" if ext in VIDEO_EXTS else "image"
            
            # 假设推理结果已获得
            # cursor.execute("INSERT OR REPLACE INTO media ...")
        conn.commit()

if __name__ == "__main__":
    main()