import MySQLdb

def get_connection():
    return MySQLdb.connect(
        host="localhost", user="root", passwd="facesense",
        db="facesense", port=3306
    )

def create_session(name):
    c = get_connection()
    cur = c.cursor()
    cur.execute("UPDATE sessions SET is_active=0 WHERE is_active=1")
    cur.execute("INSERT INTO sessions (session_name) VALUES (%s)", (name,))
    sid = cur.lastrowid
    c.commit(); c.close()
    return sid

def end_active_session():
    c = get_connection()
    cur = c.cursor()
    cur.execute("UPDATE sessions SET is_active=0, end_time=NOW() WHERE is_active=1")
    c.commit(); c.close()

def get_active_session():
    c = get_connection()
    cur = c.cursor()
    cur.execute(
        "SELECT id, session_name FROM sessions "
        "WHERE is_active=1 ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    c.close()
    return row

def log_emotion(expression, confidence, bbox, session_ref_id=None):
    try:
        c = get_connection()
        cur = c.cursor()
        x1, y1, x2, y2 = map(int, bbox)

        if session_ref_id is None:
            s = get_active_session()
            session_ref_id = s[0] if s else None

        cur.execute(
            "INSERT INTO emotion_logs "
            "(expression, confidence, x1, y1, x2, y2, session_ref_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (expression, float(confidence), x1, y1, x2, y2, session_ref_id)
        )
        c.commit(); c.close()
    except Exception as e:
        print("DB ERROR:", e)
