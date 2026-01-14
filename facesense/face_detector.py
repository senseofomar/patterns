import cv2
from pathlib import Path

CASCADE = cv2.CascadeClassifier(
    str(Path(__file__).resolve().parents[1] / "assets" / "haarcascade_frontalface_default.xml")
)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(70, 70))
