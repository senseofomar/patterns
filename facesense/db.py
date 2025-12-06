import mysql.connector

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="facesense",
        database="facesense"
    )

def log_emotion(expr, conf, bbox):
    try:
        conn = get_connection()
        cur  = conn.cursor()

        x1, y1, x2, y2 = bbox

        sql = "INSERT INTO emotion_logs (expression, confidence, x1, y1, x2, y2) VALUES (%s, %s, %s, %s, %s, %s)"
        cur.execute(sql, (expr, conf, x1, y1, x2, y2))

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB ERROR:", e)
