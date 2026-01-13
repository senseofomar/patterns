from deepface import DeepFace

def analyze_emotion(face_roi):
    r = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)[0]
    e = r["dominant_emotion"]
    return e, r["emotion"][e] / 100.0
