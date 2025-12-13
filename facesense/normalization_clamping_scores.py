import numpy as np

def mouth_based_label(left_mouth, right_mouth, upper_lip, lower_lip, face_w, face_h):
    # 1) measure
    mouth_w = np.linalg.norm(left_mouth - right_mouth)         # pixels
    lip_gap = lower_lip[1] - upper_lip[1]                     # pixels (vertical)

    # 2) normalize
    mouth_w_n = mouth_w / max(face_w, 1)
    lip_gap_n = lip_gap / max(face_h, 1)

    # 3) clamp
    mouth_w_n = max(0.0, mouth_w_n)
    lip_gap_n = max(0.0, lip_gap_n)

    # 4) score
    happy_score = 2.5 * mouth_w_n + 3.0 * lip_gap_n

    # 5) guard
    if lip_gap_n < 0.02 and mouth_w_n < 0.2:
        happy_score = 0.0

    # 6) choose
    if happy_score < 0.15:
        return "Neutral", 0.0
    else:
        # compute a crude confidence
        confidence = min(1.0, happy_score / (happy_score + 0.5))
        return "Happy", round(confidence, 2)
