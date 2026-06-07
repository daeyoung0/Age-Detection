# Age Detection Consumer Analytics

매장 카메라에서 얼굴을 감지해 연령대와 성별을 추정하고, POS 결제 데이터와 연결해 소비자 분석 대시보드를 제공하는 프로젝트입니다.

## 주요 기능

- 실시간 카메라 기반 얼굴 감지 및 연령대 추정
- 사전학습 얼굴 속성 모델 기반 성별 추정
- POS 결제 처리 및 구매 이력 저장
- 사용자, 상품, 거래 테이블 기반 통계 집계
- Next.js 기반 POS 화면과 통계 대시보드
- POS 화면에서 상품 수량 선택 지원

## 기술 스택

- Frontend: Next.js
- Backend: FastAPI
- DB: SQLite
- Vision: PyTorch, OpenCV, RetinaFace, FairFace

## 화면 구성

- `/` 홈
- `/pos` POS 결제 화면
- `/stats` 통계 대시보드

## 실행 방법

### 1. 백엔드 실행

```powershell
cd "C:\Users\ohjun\OneDrive\바탕 화면\Age-Detection-main"
python -m uvicorn backend_server:app --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 실행

```powershell
cd "C:\Users\ohjun\OneDrive\바탕 화면\Age-Detection-main"
npm run dev
```

### 3. 카메라 추론 실행

로컬 웹캠을 쓰는 경우:

```powershell
cd "C:\Users\ohjun\OneDrive\바탕 화면\Age-Detection-main"
python camera_infer_swin.py --disable-tta --debug
```

IP 카메라를 쓰는 경우:

```powershell
cd "C:\Users\ohjun\OneDrive\바탕 화면\Age-Detection-main"
python camera_infer_swin.py --camera-url http://192.168.219.101:8050/video --disable-tta --debug
```

## API

- `POST /api/camera`
- `POST /api/pay`
- `GET /api/stats`
- `GET /api/users`
- `GET /api/products`
- `GET /api/transactions`

## DB 테이블

### Users

- `user_id`
- `gender`
- `age`
- `age_group`
- `signup_date`
- `region`
- `membership_grade`

### Products

- `product_id`
- `product_name`
- `category`
- `brand`
- `price`

### Transactions

- `transaction_id`
- `user_id`
- `product_id`
- `quantity`
- `price`
- `total_amount`
- `purchase_time`
- `payment_method`

## 현재 상태

- POS 화면에서 상품 선택과 수량 선택 가능
- 통계 페이지는 카드, 차트, 표 형태의 대시보드로 구성
- 프론트 배경은 흰색 계열로 정리됨

## 참고

- 성별 모델은 `FairFace` 기반 pretrained 모델을 사용합니다.
- 얼굴이 너무 작거나 품질이 낮으면 추론 결과를 버릴 수 있습니다.
- IP 카메라 주소는 실제 장치가 노출하는 스트림 URL과 일치해야 합니다.
