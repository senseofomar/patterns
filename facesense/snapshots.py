import os, cv2, time
from datetime import datetime

def save_snapshot(frame, tag="last"):
    folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "snapshots"))
    os.makedirs(folder, exist_ok=True)

    name = "last_frame.jpg" if tag == "last" else f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    final = os.path.join(folder, name)
    tmp = final.replace(".jpg", f".{os.getpid()}.tmp.jpg")

    if not cv2.imwrite(tmp, frame):
        raise RuntimeError(f"Failed to save snapshot to {tmp}")

    for _ in range(5):
        try:
            os.replace(tmp, final)
            return final
        except PermissionError:
            time.sleep(0.05)

    os.replace(tmp, final)
    return final
