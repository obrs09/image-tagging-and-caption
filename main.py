from data_io.file_scanner import scan_files
from data_io.image_loader import load_image
from tagging.wd14_tagger import WD14Tagger
from tagging.florence_caption import FlorenceCaptioner
from tagging.joytag_z3de import JoyZ3DETagger
from tagging.joy_caption import JoyCaptioner
from embedding.embedder import ImageEmbedder
from storage.database import get_db
from storage.writer import write_image
from utils.video import sample_frames
from utils.batch_load import batch, load_images_batches
from tqdm import tqdm
from config import *
import warnings, os

warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")

wd = WD14Tagger(WD14_MODEL)
# fl = FlorenceCaptioner(FLORENCE_MODEL)
emb = ImageEmbedder(EMBEDDING_MODEL)
jz = JoyZ3DETagger(JOYTAG_MODEL, Z3DE_MODEL)
jc = JoyCaptioner(JOYCAPTION_MODEL)


def main():
    conn = get_db()
    all_files = []
    # supported_exts = IMAGE_EXT | VIDEO_EXT | GIF_EXT
    supported_exts = IMAGE_EXT
    for root, _, files in os.walk(IMG_PATH):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                all_files.append(os.path.join(root, f))

    for i in tqdm(range(0, len(all_files), BATCH_SIZE)):
        paths = all_files[i:i+BATCH_SIZE]
        imgs = []
        valid_paths = []
        
        for p in paths:
            img = load_image(p)
            if img:
                imgs.append(img)
                valid_paths.append(p)
        
        if not imgs: continue

        # wd_results = [wd.tag(img) for img in imgs]
        # jz_results = [jz.tag(img)["all_tags"] for img in imgs]
        wd_results = wd.tag_batch(imgs)  # 批处理
        jz_results = jz.tag_batch(imgs)  # 批处理
        
        # 提取 all_tags
        jz_all_tags = [result["all_tags"] for result in jz_results]
        
        tags_results = [set(wd_tags) | set(jz_tags)
                        for wd_tags, jz_tags in zip(wd_results, jz_all_tags)]
        emb_results = [emb.embed(img) for img in imgs]
        # jc_results = [jc.predict(img) for img in imgs]
        jc_results = jc.predict_batch(imgs) 
        img_w = [img.width for img in imgs]
        img_h = [img.height for img in imgs]
        img_formats = [os.path.splitext(p)[1] for p in valid_paths]  # 修正：使用 valid_paths

        for path, w, h, format, tags, embedding, caption in zip(paths, img_w, img_h, img_formats, tags_results, emb_results, jc_results):
            write_image(conn, {
            'path': path,
            'sha256': 'TODO',
            'w': w,
            'h': h,
            'format': format,
            'tags': ','.join(sorted(tags)),
            'caption': caption
            }, embedding)

# def main():
#     conn = get_db()
#     files = list(scan_files(IMG_PATH))
#     for path in tqdm(files, desc="Processing files"):
#         if path.suffix.lower() in IMAGE_EXT:
#             img = load_image(path)
#             wd_tags = wd.tag(img)
#             jz_tags = jz.tag(img)["all_tags"]
#             tags = set(wd_tags) | set(jz_tags)
#             # caption = fl.caption(img)
#             vec = emb.embed(img)
#             caption = jc.predict(img) 

#             write_image(conn, {
#                 "path": str(path),
#                 "sha256": "TODO",
#                 "w": img.width,
#                 "h": img.height,
#                 "format": path.suffix,
#                 "tags": ",".join(tags),
#                 "caption": caption
#             }, vec)

#         elif path.suffix.lower() in VIDEO_EXT:
#             frames = sample_frames(path)

if __name__ == "__main__":
    main()