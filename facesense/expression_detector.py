import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FaceSense:
    """
    Minimal emotion detector using Mediapipe FaceMesh.

    get_expression(frame) ->
        smooth_expression (str or None),
        confidence (float or None),
        bbox (x1, y1, x2, y2) or None
    """

    def __init__(self, history_size=5):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        self.history = deque(maxlen=history_size)

    def get_expression(self, frame):
        # 1) BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2) Run FaceMesh
        results = self.face_mesh.process(rgb)

        # 3) Guard: no face detected
        if not results.multi_face_landmarks:
            return None, None, None

        # 4) Use first detected face
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # 5) Landmarks -> pixel coordinates
        points = np.array([
            (int(lm.x * w), int(lm.y * h))
            for lm in landmarks.landmark
        ])

        # 6) Bounding box from all points
        xs = points[:, 0]
        ys = points[:, 1]
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # 7) Extract key landmarks
        left_mouth  = points[61]
        right_mouth = points[291]
        upper_lip   = points[13]
        lower_lip   = points[14]

        left_brow   = points[70]
        right_brow  = points[300]

        # 8) Compute features
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        lip_gap     = lower_lip[1] - upper_lip[1]
        curve       = self.mouth_curvature(left_mouth, right_mouth, upper_lip)
        brow_drop   = ((left_brow[1] + right_brow[1]) / 2) - upper_lip[1]

        # 9) Score emotions (simple heuristic)
        happy_score = (mouth_width * 0.02) + (lip_gap * 0.04) + (curve * 0.3)
        sad_score   = (-curve * 0.4)
        angry_score = (-brow_drop * 0.4)

        scores = {
            "Happy": max(happy_score, 0),
            "Sad":   max(sad_score,   0),
            "Angry": max(angry_score, 0),
            "Neutral": 0.3,  # baseline
        }

        # 10) Choose best label + confidence
        expression = max(scores, key=scores.get)
        confidence = scores[expression] / 5.0
        confidence = min(max(confidence, 0.0), 1.0)
        confidence = round(confidence, 2)

        # 11) Temporal smoothing
        self.history.append(expression)
        smooth_expression = max(set(self.history), key=self.history.count)

        return smooth_expression, confidence, bbox

    def mouth_curvature(self, left, right, top):
        """Positive = smile, negative = frown."""
        mid_y = (left[1] + right[1]) / 2
        return mid_y - top[1]
