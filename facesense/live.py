import cv2
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion


def init_camera():
    for i in (1, 0):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap
    raise IOError("Cannot open webcam")


def main():
    cap = init_camera()
    frame_count = 0
    last_emotion = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        for x, y, w, h in detect_faces(frame):
            face = frame[y:y+h, x:x+w]

            if frame_count % 5 == 0:
                try:
                    emotion, conf = analyze_emotion(face)
                    last_emotion = emotion
                    log_emotion(emotion, conf, (x, y, x+w, y+h), "webcam")
                except Exception:
                    pass

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if last_emotion:
                cv2.putText(frame, last_emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        save_snapshot(frame)
        cv2.imshow("FaceSense – Live (Mirror View)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
