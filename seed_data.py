"""
더미 데이터 삽입 스크립트
실행: python seed_data.py
"""
import os
import random
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# ── DB 경로 (backend_server.py 와 동일 로직) ──────────────────
override = os.environ.get("AGE_DETECTION_DB_PATH")
if override:
    DB_PATH = Path(override)
else:
    db_dir = Path(tempfile.gettempdir()) / "age-detection-main"
    db_dir.mkdir(parents=True, exist_ok=True)
    DB_PATH = db_dir / "store_data.db"

print(f"DB 경로: {DB_PATH}")

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# ── 테이블 초기화 (기존 더미 데이터만 삭제) ──────────────────
cur.executescript("""
    DELETE FROM Transactions;
    DELETE FROM Users;
    DELETE FROM Products;
    DELETE FROM Camera_Log;
    DELETE FROM sqlite_sequence WHERE name IN ('Transactions','Users','Products','Camera_Log');
""")
conn.commit()

# ── 상품 정의 ────────────────────────────────────────────────
PRODUCTS = [
    ("아이스 아메리카노", "커피", "Vision Cafe",  2000),
    ("핫 아메리카노",    "커피", "Vision Cafe",  2000),
    ("카페라떼",        "커피", "Vision Cafe",  5000),
    ("바닐라라떼",      "커피", "Vision Cafe",  5500),
    ("딸기 스무디",     "음료", "Vision Cafe",  5800),
    ("망고 에이드",     "음료", "Vision Cafe",  5500),
    ("치즈 케이크",     "디저트", "Vision Bakery", 6000),
    ("초코 브라우니",   "디저트", "Vision Bakery", 5200),
    ("크로플",          "디저트", "Vision Bakery", 4800),
]

cur.executemany(
    "INSERT INTO Products (product_name, category, brand, price) VALUES (?,?,?,?)",
    PRODUCTS,
)
conn.commit()

cur.execute("SELECT product_id, product_name, price FROM Products")
product_rows = cur.fetchall()
products = [(r["product_id"], r["product_name"], r["price"]) for r in product_rows]

# ── 연령대 설정 ──────────────────────────────────────────────
AGE_GROUPS = [
    ("유아/아동 (0-12)",  "infant",       (0,  12),  0.06),
    ("청소년 (13-18)",   "teen",         (13, 18),  0.12),
    ("청년층 (19-34)",   "young_adult",  (19, 34),  0.38),
    ("중장년층 (35-49)", "middle_aged",  (35, 49),  0.28),
    ("고령층 (50+)",     "senior",       (50, 75),  0.16),
]

GENDERS       = [("male", 0.43), ("female", 0.57)]
PAYMENTS      = [("card", 0.72), ("cash", 0.28)]

# ── 시간대별 가중치 (오후 3시 피크) ───────────────────────────
HOUR_WEIGHTS = {
    9: 0.03, 10: 0.05, 11: 0.08, 12: 0.11,
    13: 0.09, 14: 0.08, 15: 0.13, 16: 0.10,
    17: 0.09, 18: 0.08, 19: 0.07, 20: 0.05,
    21: 0.04,
}
HOURS  = list(HOUR_WEIGHTS.keys())
H_W    = list(HOUR_WEIGHTS.values())

# ── 연령대별 선호 상품 가중치 ─────────────────────────────────
AGE_PRODUCT_PREF = {
    "유아/아동 (0-12)":  [0, 0, 0, 0, 1, 1, 2, 3, 2],
    "청소년 (13-18)":   [2, 1, 0, 0, 2, 2, 1, 2, 2],
    "청년층 (19-34)":   [3, 2, 3, 2, 1, 1, 2, 1, 1],
    "중장년층 (35-49)": [2, 3, 3, 2, 1, 1, 2, 1, 1],
    "고령층 (50+)":     [1, 3, 2, 1, 0, 0, 2, 1, 1],
}

def pick(pool, weights):
    total = sum(weights)
    r = random.random() * total
    acc = 0
    for item, w in zip(pool, weights):
        acc += w
        if r <= acc:
            return item
    return pool[-1]

# ── 7일치 데이터 생성 ─────────────────────────────────────────
NOW   = datetime.now()
START = NOW - timedelta(days=7)

ag_labels  = [a[0] for a in AGE_GROUPS]
ag_weights = [a[3] for a in AGE_GROUPS]
ag_ranges  = {a[0]: a[2] for a in AGE_GROUPS}

NUM_TRANSACTIONS = 280
random.seed(42)

for _ in range(NUM_TRANSACTIONS):
    # 날짜·시간
    day_offset = random.randint(0, 6)
    hour       = pick(HOURS, H_W)
    minute     = random.randint(0, 59)
    second     = random.randint(0, 59)
    ts = START + timedelta(days=day_offset, hours=hour - START.hour,
                           minutes=minute, seconds=second)
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

    # 고객
    age_group  = pick(ag_labels, ag_weights)
    age_range  = ag_ranges[age_group]
    age        = random.randint(*age_range)
    gender     = pick([g[0] for g in GENDERS], [g[1] for g in GENDERS])
    payment    = pick([p[0] for p in PAYMENTS], [p[1] for p in PAYMENTS])

    cur.execute(
        "INSERT INTO Users (gender, age, age_group, signup_date, region, membership_grade) VALUES (?,?,?,?,?,?)",
        (gender, age, age_group, ts_str, "서울", random.choice(["guest", "member", "vip"])),
    )
    user_id = cur.lastrowid

    # 상품 (연령대 선호 반영)
    pref_w = AGE_PRODUCT_PREF.get(age_group, [1]*len(products))
    pid, pname, price = pick(products, pref_w)
    quantity    = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05])[0]
    total       = price * quantity

    cur.execute(
        """INSERT INTO Transactions
           (user_id, product_id, quantity, price, total_amount, purchase_time, payment_method)
           VALUES (?,?,?,?,?,?,?)""",
        (user_id, pid, quantity, price, total, ts_str, payment),
    )

conn.commit()

# ── 결과 확인 ────────────────────────────────────────────────
cur.execute("SELECT COUNT(*) FROM Users")
print(f"Users     : {cur.fetchone()[0]}명")
cur.execute("SELECT COUNT(*) FROM Products")
print(f"Products  : {cur.fetchone()[0]}개")
cur.execute("SELECT COUNT(*) FROM Transactions")
print(f"Transactions: {cur.fetchone()[0]}건")
cur.execute("SELECT SUM(total_amount) FROM Transactions")
print(f"총 매출   : {cur.fetchone()[0]:,}원")

conn.close()
print("\n✅ 더미 데이터 삽입 완료! 백엔드를 실행한 뒤 대시보드를 확인하세요.")
