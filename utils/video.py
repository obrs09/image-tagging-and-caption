import cv2


def sample_frames(video_path, interval=2.0):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(fps * interval)


    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame[:, :, ::-1]) # BGR->RGB
        idx += 1
    cap.release()
    return frames