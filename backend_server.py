from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import os
import tempfile

app = FastAPI(title="연령대 분석 시스템 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_path() -> Path:
    override = os.environ.get("AGE_DETECTION_DB_PATH")
    if override:
        return Path(override)

    # SQLite on this Windows environment fails under the current project path,
    # so store the DB in a stable ASCII temp directory by default.
    db_dir = Path(tempfile.gettempdir()) / "age-detection-main"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "store_data.db"


DB_PATH = get_db_path()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 카메라 기록 테이블 (얼굴 크기 정보 포함)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Camera_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            age_group TEXT,
            face_size INTEGER
        )
    """)
    
    # 카메라 pos기 매칭 테이블
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS POS_Data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            item_name TEXT,
            price INTEGER,
            matched_age_group TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class CameraData(BaseModel):
    age_group: str
    face_size: int

class PaymentData(BaseModel):
    item_name: str
    price: int

LABEL_TRANSLATOR = {
    "infant": "유아/아동 (0-12)",
    "child": "유아/아동 (0-12)",
    "teen": "청소년 (13-18)",
    "young_adult": "청년층 (19-34)",
    "middle_aged": "중장년층 (35-49)",
    "senior": "고령층 (50+)"
}

@app.post("/api/camera")
def save_camera_log(data: CameraData):
    """카메라가 보내는 연령대 데이터를 DB에 저장합니다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(
        "INSERT INTO Camera_Log (timestamp, age_group, face_size) VALUES (?, ?, ?)",
        (now, data.age_group, data.face_size)
    )
    conn.commit()
    conn.close()
    
    return {"message": "카메라 데이터 저장 성공", "time": now}


@app.post("/api/pay")
def process_payment(data: PaymentData):
    """결제 직전 가장 큰 얼굴 데이터를 찾아 매칭합니다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    end_time = now.strftime("%Y-%m-%d %H:%M:%S")
    start_time = (now - timedelta(seconds=2)).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        SELECT age_group FROM Camera_Log
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC LIMIT 1
    """, (start_time, end_time))
    
    result = cursor.fetchone()
    
    if result:
        raw_age = result[0]
        matched_age = LABEL_TRANSLATOR.get(raw_age, "알 수 없음")
    else:
        matched_age = "알 수 없음 (미검출)"
  
    cursor.execute(
        "INSERT INTO POS_Data (timestamp, item_name, price, matched_age_group) VALUES (?, ?, ?, ?)",
        (now_str, data.item_name, data.price, matched_age)
    )
    conn.commit()
    conn.close()
    
    return {
        "message": "결제 및 연령대 매칭 완료",
        "item": data.item_name,
        "matched_age_group": matched_age
    }


@app.get("/api/stats")
def get_statistics():
    """매칭 데이터를 통계로 반환합니다."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT matched_age_group, COUNT(*) as count, SUM(price) as total_sales 
        FROM POS_Data 
        GROUP BY matched_age_group
    """)
    rows = cursor.fetchall()
    conn.close()
    
    stats = []
    for row in rows:
        stats.append({
            "age_group": row[0],
            "count": row[1],
            "total_sales": row[2]
        })
        
    return {"statistics": stats}
