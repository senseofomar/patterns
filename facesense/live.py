import cv2, time
from collections import deque
from facesense.core.face_detector import detect_faces
from facesense.core.emotion import analyze_emotion
from facesense.snapshots.snapshot import save_snapshot
from facesense.storage.db import log_emotion, get_active_session

COLORS = {
    'angry': (0, 0, 255), 'happy': (0, 255, 255), 'sad': (255, 0, 0),
    'neutral': (0, 255, 0), 'surprise': (255, 255, 0),
    'fear': (128, 0, 128), 'disgust': (0, 128, 0)
}

def init_camera():
    for i in (1, 0):
        cap = cv2.VideoCapture(i)
        if cap.isOpened(): return cap
    raise IOError("Cannot open webcam")

def dominant(buf):
    return max(set(buf), key=buf.count) if buf else "neutral"

def draw_hud(f, x, y, w, h, c, scan, conf):
    l, t = int(w * .15), 2
    for a,b in [((x,y),(x+l,y)),((x,y),(x,y+l)),((x+w,y),(x+w-l,y)),((x+w,y),(x+w,y+l)),
                ((x,y+h),(x+l,y+h)),((x,y+h),(x,y+h-l)),((x+w,y+h),(x+w-l,y+h)),((x+w,y+h),(x+w,y+h-l))]:
        cv2.line(f,a,b,c,t)
    sy = y + int(scan*h)
    cv2.line(f,(x,sy),(x+w,sy),(0,255,0),2)
    cv2.rectangle(f,(x,y-15),(x+w,y-10),(50,50,50),-1)
    cv2.rectangle(f,(x,y-15),(x+int(w*conf),y-10),c,-1)

def draw_stats(f, fps, rec, n):
    cv2.rectangle(f,(10,10),(220,90),(0,0,0),-1)
    cv2.rectangle(f,(10,10),(220,90),(0,255,0),1)
    cv2.putText(f,"SYSTEM: ONLINE",(20,35),cv2.FONT_HERSHEY_PLAIN,1.1,(0,255,0),1)
    cv2.putText(f,f"FPS: {int(fps)}",(20,55),cv2.FONT_HERSHEY_PLAIN,1.1,(0,255,255),1)
    if rec and n%30<15:
        cv2.circle(f,(28,70),5,(0,0,255),-1)
        cv2.putText(f,"LOGS: ACTIVE",(40,75),cv2.FONT_HERSHEY_PLAIN,1.1,(0,0,255),1)
    else:
        cv2.putText(f,"LOGS: IDLE",(20,75),cv2.FONT_HERSHEY_PLAIN,1.1,(100,100,100),1)

def main():
    cap = init_camera()
    buf, scan, d, prev, rec = deque(maxlen=7), 0.0, .05, 0, False
    emo, conf, n = "neutral", 0.0, 0

    while True:
        ok, f = cap.read()
        if not ok: break
        f = cv2.flip(f,1); n+=1

        now=time.time(); fps=1/(now-prev) if prev else 0; prev=now
        scan+=d; d*=-1 if scan>=1 or scan<=0 else 1

        faces = detect_faces(f)
        if faces: faces=[max(faces,key=lambda r:r[2]*r[3])]

        for x,y,w,h in faces:
            roi=f[y:y+h,x:x+w]
            if n%5==0:
                if n%30==0: rec = get_active_session() is not None
                try:
                    r, c = analyze_emotion(roi)
                    buf.append(r); emo=dominant(buf); conf=c
                    log_emotion(emo,c,(x,y,x+w,y+h))
                except: pass

            col=COLORS.get(emo,(0,255,0))
            draw_hud(f,x,y,w,h,col,scan,conf)
            lab=emo.upper()
            (tw,th),_=cv2.getTextSize(lab,cv2.FONT_HERSHEY_SIMPLEX,.8,2)
            cv2.rectangle(f,(x,y-35),(x+tw+10,y),col,-1)
            cv2.putText(f,lab,(x+5,y-10),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)

        draw_stats(f,fps,rec,n)
        save_snapshot(f)
        cv2.imshow("FaceSense – Live (Mirror View)",f)
        if cv2.waitKey(1)&0xFF==ord("q"): break

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
