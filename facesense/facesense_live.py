import cv2
from expression_detector import FaceSense
from db import log_emotion


def draw_results(frame, bbox, label, confidence):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence*100:.0f}%)",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2)


def main():
    detector = FaceSense()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        expression, confidence, bbox = detector.get_expression(frame)

        if bbox is None:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
        else:
            draw_results(frame, bbox, expression, confidence)
            log_emotion(expression, confidence, bbox)

        cv2.imshow("FaceSense Live", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
