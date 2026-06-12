from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import sqlite3
import tempfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="Age Detection API")


def get_allowed_origins() -> list[str]:
    # 기본값은 데모 편의를 위해 모든 origin 허용. 운영 배포 시 ALLOWED_ORIGINS에
    # 콤마로 구분된 origin 목록을 지정하면 해당 origin만 허용한다.
    raw = os.environ.get("ALLOWED_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


AGE_GROUP_LABELS = {
    "infant": "유아/아동 (0-12)",
    "child": "유아/아동 (0-12)",
    "teen": "청소년 (13-18)",
    "young_adult": "청년층 (19-34)",
    "middle_aged": "중장년층 (35-49)",
    "senior": "고령층 (50+)",
    "unknown": "알 수 없음",
}

AGE_GROUP_AGE_HINTS = {
    "infant": 5,
    "child": 10,
    "teen": 16,
    "young_adult": 27,
    "middle_aged": 42,
    "senior": 58,
    "unknown": None,
}


def get_db_path() -> Path:
    override = os.environ.get("AGE_DETECTION_DB_PATH")
    if override:
        return Path(override)

    db_dir = Path(tempfile.gettempdir()) / "age-detection-main"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "store_data.db"


DB_PATH = get_db_path()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str) -> None:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing = {row[1] for row in cursor.fetchall()}
    if column_name not in existing:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def init_db() -> None:
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            gender TEXT,
            age INTEGER,
            age_group TEXT,
            signup_date DATETIME,
            region TEXT,
            membership_grade TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Products (
            product_id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT NOT NULL,
            category TEXT,
            brand TEXT,
            price INTEGER NOT NULL,
            UNIQUE(product_name, brand, price)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price INTEGER NOT NULL,
            total_amount INTEGER NOT NULL,
            purchase_time DATETIME NOT NULL,
            payment_method TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES Users(user_id),
            FOREIGN KEY (product_id) REFERENCES Products(product_id)
        )
        """
    )

    # Camera logs remain as a short-lived matching buffer for POS purchases.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Camera_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            age_group TEXT,
            age INTEGER,
            face_size INTEGER,
            gender TEXT
        )
        """
    )

    ensure_column(conn, "Users", "gender", "TEXT")
    ensure_column(conn, "Users", "age", "INTEGER")
    ensure_column(conn, "Users", "age_group", "TEXT")
    ensure_column(conn, "Users", "signup_date", "DATETIME")
    ensure_column(conn, "Users", "region", "TEXT")
    ensure_column(conn, "Users", "membership_grade", "TEXT")

    ensure_column(conn, "Products", "category", "TEXT")
    ensure_column(conn, "Products", "brand", "TEXT")
    ensure_column(conn, "Products", "price", "INTEGER")

    ensure_column(conn, "Transactions", "quantity", "INTEGER")
    ensure_column(conn, "Transactions", "price", "INTEGER")
    ensure_column(conn, "Transactions", "total_amount", "INTEGER")
    ensure_column(conn, "Transactions", "purchase_time", "DATETIME")
    ensure_column(conn, "Transactions", "payment_method", "TEXT")

    ensure_column(conn, "Camera_Log", "age_group", "TEXT")
    ensure_column(conn, "Camera_Log", "age", "INTEGER")
    ensure_column(conn, "Camera_Log", "face_size", "INTEGER")
    ensure_column(conn, "Camera_Log", "gender", "TEXT")

    conn.commit()
    conn.close()


init_db()


class CameraData(BaseModel):
    age_group: str
    face_size: int
    gender: str = "unknown"
    age: int | None = None


class PaymentData(BaseModel):
    item_name: str
    price: int
    quantity: int = 1
    payment_method: str = "unknown"
    category: str = "unknown"
    brand: str = "unknown"


def normalize_age_group(raw_age_group: str | None) -> str:
    if not raw_age_group:
        return "unknown"
    return raw_age_group if raw_age_group in AGE_GROUP_LABELS else "unknown"


def to_display_age_group(raw_age_group: str | None) -> str:
    return AGE_GROUP_LABELS[normalize_age_group(raw_age_group)]


def infer_age(age_group: str, explicit_age: int | None) -> int | None:
    if explicit_age is not None:
        return explicit_age
    return AGE_GROUP_AGE_HINTS.get(normalize_age_group(age_group))


def create_user_from_camera(cursor: sqlite3.Cursor, age_group: str, gender: str, age: int | None) -> int:
    signup_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO Users (gender, age, age_group, signup_date, region, membership_grade)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            gender or "unknown",
            age,
            to_display_age_group(age_group),
            signup_date,
            "unknown",
            "guest",
        ),
    )
    return int(cursor.lastrowid)


def ensure_product(cursor: sqlite3.Cursor, item_name: str, price: int, category: str, brand: str) -> int:
    cursor.execute(
        """
        SELECT product_id FROM Products
        WHERE product_name = ? AND brand = ? AND price = ?
        """,
        (item_name, brand, price),
    )
    existing = cursor.fetchone()
    if existing:
        return int(existing["product_id"])

    cursor.execute(
        """
        INSERT INTO Products (product_name, category, brand, price)
        VALUES (?, ?, ?, ?)
        """,
        (item_name, category, brand, price),
    )
    return int(cursor.lastrowid)


def get_recent_camera_match(cursor: sqlite3.Cursor, seconds: int = 2) -> sqlite3.Row | None:
    now = datetime.now()
    end_time = now.strftime("%Y-%m-%d %H:%M:%S")
    start_time = (now - timedelta(seconds=seconds)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        SELECT age_group, age, gender, face_size, timestamp
        FROM Camera_Log
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (start_time, end_time),
    )
    return cursor.fetchone()


@app.post("/api/camera")
def save_camera_log(data: CameraData):
    conn = get_conn()
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    age_group = normalize_age_group(data.age_group)
    age = infer_age(age_group, data.age)

    cursor.execute(
        """
        INSERT INTO Camera_Log (timestamp, age_group, age, face_size, gender)
        VALUES (?, ?, ?, ?, ?)
        """,
        (now, age_group, age, data.face_size, data.gender),
    )
    conn.commit()
    conn.close()

    return {"message": "camera data saved", "time": now}


@app.post("/api/pay")
def process_payment(data: PaymentData):
    conn = get_conn()
    cursor = conn.cursor()

    matched = get_recent_camera_match(cursor)
    raw_age_group = matched["age_group"] if matched else "unknown"
    matched_gender = matched["gender"] if matched else "unknown"
    matched_age = matched["age"] if matched else None

    user_id = create_user_from_camera(
        cursor,
        age_group=raw_age_group,
        gender=matched_gender,
        age=matched_age,
    )
    product_id = ensure_product(
        cursor,
        item_name=data.item_name,
        price=data.price,
        category=data.category,
        brand=data.brand,
    )

    quantity = max(1, int(data.quantity))
    total_amount = int(data.price) * quantity
    purchase_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute(
        """
        INSERT INTO Transactions (
            user_id, product_id, quantity, price, total_amount, purchase_time, payment_method
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            product_id,
            quantity,
            int(data.price),
            total_amount,
            purchase_time,
            data.payment_method,
        ),
    )
    transaction_id = int(cursor.lastrowid)

    conn.commit()
    conn.close()

    return {
        "message": "payment matched",
        "transaction_id": transaction_id,
        "user_id": user_id,
        "product_id": product_id,
        "item": data.item_name,
        "quantity": quantity,
        "payment_method": data.payment_method,
        "matched_age_group": to_display_age_group(raw_age_group),
        "matched_gender": matched_gender or "unknown",
        "matched_age": matched_age,
    }


@app.get("/api/stats")
def get_statistics():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            u.age_group AS age_group,
            u.gender AS gender,
            COUNT(t.transaction_id) AS count,
            SUM(t.total_amount) AS total_sales
        FROM Transactions t
        JOIN Users u ON u.user_id = t.user_id
        GROUP BY u.age_group, u.gender
        ORDER BY total_sales DESC, count DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    return {
        "statistics": [
            {
                "age_group": row["age_group"],
                "gender": row["gender"],
                "count": row["count"],
                "total_sales": row["total_sales"],
            }
            for row in rows
        ]
    }


@app.get("/api/users")
def get_users():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT user_id, gender, age, age_group, signup_date, region, membership_grade
        FROM Users
        ORDER BY user_id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return {"users": [dict(row) for row in rows]}


@app.get("/api/products")
def get_products():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT product_id, product_name, category, brand, price
        FROM Products
        ORDER BY product_id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return {"products": [dict(row) for row in rows]}


@app.get("/api/transactions")
def get_transactions():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            t.transaction_id,
            t.user_id,
            t.product_id,
            p.product_name,
            t.quantity,
            t.price,
            t.total_amount,
            t.purchase_time,
            t.payment_method
        FROM Transactions t
        JOIN Products p ON p.product_id = t.product_id
        ORDER BY t.transaction_id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return {"transactions": [dict(row) for row in rows]}
