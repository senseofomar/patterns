import cv2
from pathlib import Path
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.storage.db import log_emotion

def run_on_image(image_path, show=True, log_to_db=True):
    p = Path(image_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[2] / p

    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Invalid image path: {p}")

    faces = detect_faces(img)
    print(f"Faces detected: {len(faces)}")

    for x,y,w,h in faces:
        emo, conf = analyze_emotion(img[y:y+h, x:x+w])
        if log_to_db:
            log_emotion(emo, conf, (x,y,x+w,y+h))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, emo, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)

    out_dir = Path(__file__).resolve().parents[1] / "storage" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{p.stem}_processed.jpg"
    cv2.imwrite(str(out), img)
    print(f"✅ Processed image saved to: {out}")

    if show:
        cv2.imshow("FaceSense – Static", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_on_image("facesense/storage/raw/gg.png")
