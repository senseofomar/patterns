import sys, os, time, cv2, numpy as np
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from facesense.storage.db import (
    create_session, end_active_session, get_active_session, get_connection
)
from facesense.core.face_detector import detect_faces

SNAPSHOT = os.path.join(os.getcwd(), "snapshots", "last_frame.jpg")
_last = None

def load_snapshot():
    global _last
    if not os.path.exists(SNAPSHOT):
        return _last
    for _ in range(3):
        img = cv2.imread(SNAPSHOT)
        if img is not None:
            _last = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return _last
        time.sleep(0.01)
    return _last

def get_session_data(sid):
    c = get_connection()
    df = pd.read_sql(
        "SELECT ts, expression, confidence FROM emotion_logs "
        "WHERE session_ref_id=%s ORDER BY ts", c, params=(sid,)
    )
    c.close()
    return df

st.set_page_config("FaceSense AI", layout="wide", page_icon="🧠")

st.sidebar.title("🎛 Control Panel")
mode = st.sidebar.radio("Select Mode", ["📡 Live Monitor","🖼️ Static Forensics","📂 Session History"])

active = get_active_session()
if active:
    st.sidebar.success(f"🔴 RECORDING: {active[1]}")
    if st.sidebar.button("Stop Session"): end_active_session(); st.rerun()
else:
    name = st.sidebar.text_input("New Session Name")
    if st.sidebar.button("Start Recording") and name:
        create_session(name); st.rerun()

# ---------- LIVE ----------
if mode == "📡 Live Monitor":
    c1,c2 = st.columns([2,1])
    img_spot = c1.empty()
    met_spot = c2.empty()
    chart_spot = c2.empty()
    last = 0

    while True:
        img = load_snapshot()
        if img is not None:
            img_spot.image(img, use_container_width=True,
                           caption=datetime.now().strftime("%H:%M:%S"))
        else:
            img_spot.warning("Waiting for camera...")

        if time.time() - last > 1 and active:
            df = get_session_data(active[0])
            if not df.empty:
                row = df.iloc[-1]
                m1,m2 = met_spot.columns(2)
                m1.metric("Emotion", row.expression.upper())
                m2.metric("Confidence", f"{row.confidence*100:.1f}%")

                chart = alt.Chart(df).mark_tick(thickness=3).encode(
                    x='ts:T', y='expression:N', color='expression:N'
                ).properties(height=250)
                chart_spot.altair_chart(chart, use_container_width=True)
            last = time.time()
        time.sleep(0.05)

# ---------- STATIC ----------
elif mode == "🖼️ Static Forensics":
    f = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if f:
        data = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if st.button("Analyze"):
            from facesense.core.emotion import analyze_emotion
            faces = detect_faces(img)
            out = rgb.copy()

            for x,y,w,h in faces:
                emo, conf = analyze_emotion(img[y:y+h,x:x+w])
                cv2.rectangle(out,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(out, emo.upper(), (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

            st.image(out)
            st.success(f"{len(faces)} face(s) detected")

# ---------- HISTORY ----------
elif mode == "📂 Session History":
    c = get_connection()
    sessions = pd.read_sql("SELECT id, session_name FROM sessions", c)
    if not sessions.empty:
        sel = st.selectbox("Session", sessions.session_name)
        sid = sessions[sessions.session_name==sel].id.values[0]
        if st.button("Generate Report"):
            df = pd.read_sql(
                "SELECT ts, expression, confidence FROM emotion_logs "
                "WHERE session_ref_id=%s ORDER BY ts", c, params=(sid,)
            )
            if not df.empty:
                st.metric("Dominant", df.expression.mode()[0])
                st.metric("Avg Confidence", f"{df.confidence.mean()*100:.1f}%")
                st.altair_chart(
                    alt.Chart(df).mark_tick().encode(
                        x='ts:T', y='expression:N', color='expression:N'
                    ),
                    use_container_width=True
                )
    c.close()
