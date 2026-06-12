# 소비자 분석 시스템 (Age Detection Consumer Analytics)

> **제출 정보** *(아래 항목을 채워 주세요)*
> - 과목명: _______________
> - 학번 / 이름: _______________
> - 제출일: _______________

매장 카메라로 손님 얼굴을 감지해 **연령대·성별을 실시간 추정**하고, 그 결과를 **POS 결제 데이터와 자동으로 연결**하여 소비 패턴을 분석하는 대시보드 시스템입니다.

> 카메라가 인식한 손님의 연령·성별 → 결제 시점 기준 **최근 2초 내 인식 결과와 매칭** → 누가(연령/성별) 무엇을 얼마에 샀는지를 통계로 집계합니다.

---

## 목차

1. [주요 기능](#주요-기능)
2. [시스템 구조](#시스템-구조)
3. [기술 스택](#기술-스택)
4. [프로젝트 구조](#프로젝트-구조)
5. [모델 구성](#모델-구성)
6. [사전 준비 및 설치](#사전-준비-및-설치)
7. [실행 방법](#실행-방법)
8. [API 명세](#api-명세)
9. [데이터베이스 스키마](#데이터베이스-스키마)
10. [데모 시연 가이드](#데모-시연-가이드)
11. [문제 해결](#문제-해결)
12. [모델 파일 & Git LFS](#모델-파일--git-lfs)

---

## 주요 기능

- **실시간 얼굴 감지** — RetinaFace(ResNet-50)로 프레임에서 가장 크고 유효한 얼굴 1개를 선택
- **연령대 추정** — Swin/ConvNeXt 기반 앙상블 모델로 6개 연령대 분류 (`infant`/`child`/`teen`/`young_adult`/`middle_aged`/`senior`)
- **성별 추정** — FairFace(ResNet-34) pretrained 모델 기반
- **추정 안정화** — 최근 프레임 확률 평균(temporal smoothing), 저품질 얼굴(흐림/저조도/저대비) 자동 폐기, 연령 편향 보정
- **POS 결제 ↔ 카메라 매칭** — 결제 시점 기준 최근 2초 내 카메라 인식 결과를 자동 연결
- **통계 대시보드** — 연령대·성별 매출, 고객/상품/거래 이력을 카드·차트·표로 시각화
- **더미 데이터 시드** — 시연 전 7일치 가상 거래 데이터 자동 생성

---

## 시스템 구조

실행 방식은 두 가지입니다.

### 모드 A — 로컬 단독 실행 (간단, 권장 데모)

추론·카메라·POS·통계를 **모두 한 PC에서** 실행합니다. GPU가 있으면 빠르고, 없으면 CPU로도 동작합니다.

```
 [웹캠 / IP 카메라]
       │
       ▼
 [camera_infer_swin.py]  ── RetinaFace + 연령앙상블 + FairFace 추론
       │
       │  POST /api/camera (연령/성별)
       ▼
 [backend_server.py :8000]  ──  SQLite DB  ──  [프론트 :3000]
       ▲                                          (POS / 통계 화면)
       └── POST /api/pay (결제, 최근 2초 카메라 결과와 매칭)
```

### 모드 B — RunPod GPU 분산 실행 (무거운 추론을 외부 GPU로)

카메라·화면·POS·통계는 내 PC에서, **무거운 추론만 RunPod GPU**에서 실행합니다. 자세한 단계는 [`실행가이드.md`](./실행가이드.md) 참고.

```
 [폰 IP카메라] ──▶ [내 PC: camera_client.py] ──JPEG──▶ [Cloudflare 터널] ──▶ [RunPod: infer_server.py (GPU)]
                          │   ◀──────────────── age/gender 결과 ────────────────────┘
                          ├─▶ 영상 창에 박스 + 라벨
                          └─▶ POST /api/camera ─▶ [내 PC: backend_server.py :8000] ◀─ [프론트 :3000]
```

| 위치 | 프로그램 | 포트 | 역할 |
|------|----------|------|------|
| RunPod (GPU) | `infer_server.py` | 8000 | 무거운 추론(RetinaFace+연령앙상블+FairFace) |
| 내 PC | `backend_server.py` | 8000 | DB 저장, POS/통계 API |
| 내 PC | `npm run dev` | 3000 | POS 화면, 통계 대시보드 |
| 내 PC | `camera_client.py` | — | 카메라 캡처 + 화면 + 추론 요청 |

---

## 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| Frontend | Next.js 16, React 19 |
| Backend | FastAPI, Uvicorn, Pydantic |
| Database | SQLite |
| 얼굴 감지 | RetinaFace (resnet50_2020-07-20) |
| 연령 모델 | PyTorch, timm (Swin Transformer / ConvNeXt 앙상블) |
| 성별 모델 | FairFace (ResNet-34) |
| 영상 처리 | OpenCV, Pillow, NumPy |
| GPU 배포(선택) | RunPod, runpodctl, Cloudflare Tunnel |

---

## 프로젝트 구조

```
Age-Detection-demo/
├── app/                       # Next.js 프론트엔드
│   ├── page.js                #   홈 (바로가기)
│   ├── pos/page.js            #   POS 결제 화면
│   ├── stats/page.js          #   통계 대시보드
│   ├── components/top-nav.js  #   공용 상단 네비게이션
│   ├── lib/dashboard.js       #   대시보드 데이터 헬퍼
│   ├── layout.js
│   └── globals.css
│
├── src/swinface_age/          # 연령/성별 추론 모델 패키지
│   ├── model.py               #   Swin/VGGFace 분류기 + 앙상블 로더
│   ├── gender.py              #   FairFace 성별 분류기
│   └── dataset.py             #   전처리 / TTA 변환
│
├── backend_server.py          # 로컬 백엔드 (POS/통계 API + SQLite)
├── infer_server.py            # RunPod GPU 추론 API 서버 (모드 B)
├── camera_infer_swin.py       # 로컬 단독 추론 스크립트 (모드 A)
├── camera_client.py           # 카메라 클라이언트 (모드 B, 원격 GPU 추론)
├── pipeline_swin.py           # 추론 파이프라인 유틸
├── preview_camera.py          # 추론 없이 카메라 미리보기 (진단용)
├── seed_data.py               # 더미 거래 데이터 생성
├── smoke_test_camera.py       # 카메라 스모크 테스트
│
├── fair_face_models/          # FairFace pretrained 가중치
├── ensemble_korean.pt         # 연령 앙상블 모델 (Git LFS, ~786MB)
│
├── requirements.txt           # Python 의존성
├── package.json               # Node 의존성
├── 실행가이드.md               # RunPod GPU 모드 상세 실행 가이드
└── RUNPOD.md                  # RunPod 배포 메모
```

---

## 모델 구성

추론 파이프라인은 세 단계로 구성됩니다.

1. **얼굴 감지 — RetinaFace (ResNet-50)**
   프레임에서 얼굴 박스와 5점 랜드마크를 검출합니다. 점수가 낮거나(`< 0.8`), 너무 작거나(`< 80px`), 비정상적으로 가로로 넓은 박스는 폐기하고 가장 큰 유효 얼굴 하나만 사용합니다.

2. **연령대 분류 — 앙상블 (`ensemble_korean.pt`)**
   여러 백본(Swin Transformer / ConvNeXt)의 softmax 확률을 가중 평균해 6개 연령대로 분류합니다. timm 버전 차이에 대비한 호환 로딩(`_load_state_dict_compat`)을 포함합니다.
   - 학습 데이터의 느슨한 크롭과 맞추기 위해 얼굴 박스를 15% 확장
   - 최근 프레임 확률 평균으로 단일 오인식 프레임의 영향을 완화
   - 성별에 따른 연령 편향 보정(`apply_age_bias`)으로 20대가 청소년/아동으로 쏠리는 현상 완화

3. **성별 추정 — FairFace (ResNet-34)**
   랜드마크로 정렬한 얼굴 크롭을 입력해 성별을 추정합니다(공식 FairFace 체크포인트, 로짓 인덱스 7:9 사용).

---

## 사전 준비 및 설치

### 요구 사항

- Python 3.10+ (개발 환경: 3.11)
- Node.js 18+
- (선택) CUDA 12.1 지원 GPU — 없으면 CPU로 동작
- 웹캠 또는 IP 카메라(폰 IP Webcam 앱 등)

### 1) Python 의존성

```powershell
# GPU (CUDA 12.1, 권장)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# 또는 CPU 전용
# pip install torch==2.5.1 torchvision==0.20.1

pip install -r requirements.txt
```

### 2) Node 의존성

```powershell
npm install
```

### 3) 모델 파일 내려받기 (Git LFS)

```powershell
git lfs install
git lfs pull   # ensemble_korean.pt (~786MB) 다운로드
```

### 4) (선택) 환경변수 설정

기본값으로도 동작하지만, 필요하면 `.env.example`를 복사해 값을 조정합니다.

```powershell
Copy-Item .env.example .env
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ALLOWED_ORIGINS` | `*` | CORS 허용 origin(콤마 구분). 운영 시 프론트 주소로 제한 |
| `AGE_DETECTION_DB_PATH` | `%TEMP%/age-detection-main/store_data.db` | SQLite DB 파일 경로 |

---

## 실행 방법

### 모드 A — 로컬 단독 실행

세 개의 터미널을 사용합니다.

**터미널 1 — 백엔드**
```powershell
python -m uvicorn backend_server:app --host 0.0.0.0 --port 8000
```

**터미널 2 — 프론트엔드**
```powershell
npm run dev      # http://localhost:3000
```

**터미널 3 — 카메라 추론**
```powershell
# 로컬 웹캠
python camera_infer_swin.py --disable-tta --debug

# IP 카메라
python camera_infer_swin.py --camera-url http://192.168.219.101:8050/video --disable-tta --debug
```

영상 창에 초록 박스와 `young_adult / male (0.86)` 형태의 라벨이 뜨면 정상입니다. 종료는 영상 창에서 `q`.

#### 주요 옵션 (`camera_infer_swin.py`)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--camera-url` | (빈값) | IP 카메라 스트림 URL. 빈값이면 로컬 웹캠 사용 |
| `--camera-index` | `0` | 로컬 카메라 인덱스 |
| `--disable-tta` | off | 테스트타임 증강 비활성화(빠름) |
| `--face-margin` | `0.15` | 얼굴 박스 확장 비율 |
| `--min-score` | `0.8` | 얼굴 검출 최소 점수 |
| `--min-face` | `80` | 최소 얼굴 크기(px) |
| `--smooth` | `8` | 평균낼 최근 프레임 수 |
| `--disable-gender` | off | 성별 추정 끄기 |
| `--debug` | off | 클래스별 확률 출력 + 크롭을 `debug_crops/`에 저장 |

### 모드 B — RunPod GPU 분산 실행

RunPod Pod 생성, 번들 전송, Cloudflare 터널, `camera_client.py` 실행 등 전체 절차는 **[`실행가이드.md`](./실행가이드.md)** 에 단계별로 정리되어 있습니다.

---

## API 명세

베이스 URL: `http://localhost:8000`

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/camera` | 카메라 인식 결과 저장(매칭 버퍼) |
| `POST` | `/api/pay` | 결제 처리 + 최근 2초 카메라 결과와 매칭 |
| `GET` | `/api/stats` | 연령대·성별별 거래 수/매출 집계 |
| `GET` | `/api/users` | 사용자 목록 |
| `GET` | `/api/products` | 상품 목록 |
| `GET` | `/api/transactions` | 거래 이력(상품명 조인) |

### 요청 예시

**`POST /api/camera`**
```json
{ "age_group": "young_adult", "face_size": 24000, "gender": "male", "age": 27 }
```

**`POST /api/pay`**
```json
{ "item_name": "아이스 아메리카노", "price": 2000, "quantity": 1,
  "payment_method": "card", "category": "커피", "brand": "Vision Cafe" }
```

응답에는 결제로 생성된 `transaction_id`, `user_id`와 함께 매칭된 `matched_age_group`, `matched_gender`, `matched_age`가 포함됩니다. 결제 시점에 카메라가 얼굴을 잡고 있지 않았으면 `unknown`으로 기록됩니다.

---

## 데이터베이스 스키마

SQLite 파일은 기본적으로 임시 디렉터리(`%TEMP%/age-detection-main/store_data.db`)에 생성됩니다. 환경변수 `AGE_DETECTION_DB_PATH`로 경로를 지정할 수 있습니다.

**Users** — 결제 시점에 카메라 매칭 결과로부터 생성되는 고객 레코드

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `user_id` | INTEGER PK | 자동 증가 |
| `gender` | TEXT | 성별 |
| `age` | INTEGER | 추정 나이(연령대 힌트값) |
| `age_group` | TEXT | 표시용 연령대 라벨 |
| `signup_date` | DATETIME | 생성 시각 |
| `region` | TEXT | 지역 |
| `membership_grade` | TEXT | 회원 등급 |

**Products** — 상품 (이름/브랜드/가격 조합이 유일)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `product_id` | INTEGER PK | 자동 증가 |
| `product_name` | TEXT | 상품명 |
| `category` | TEXT | 카테고리 |
| `brand` | TEXT | 브랜드 |
| `price` | INTEGER | 단가 |

**Transactions** — 거래

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `transaction_id` | INTEGER PK | 자동 증가 |
| `user_id` | INTEGER FK | → Users |
| `product_id` | INTEGER FK | → Products |
| `quantity` | INTEGER | 수량 |
| `price` | INTEGER | 단가 |
| `total_amount` | INTEGER | 합계(price × quantity) |
| `purchase_time` | DATETIME | 결제 시각 |
| `payment_method` | TEXT | 결제 수단 |

**Camera_Log** — POS 매칭용 단기 버퍼 (`timestamp`, `age_group`, `age`, `face_size`, `gender`)

---

## 데모 시연 가이드

1. (선택) 시연 전 더미 데이터로 대시보드 채우기
   ```powershell
   python seed_data.py     # 7일치, 280건 가상 거래 생성
   ```
2. 카메라가 손님 얼굴을 비추는 상태에서
3. `http://localhost:3000/pos` → 상품 선택 → 수량 → **결제**
   - 결제 시점 **최근 2초 내** 카메라가 보낸 연령/성별과 자동 매칭됩니다.
4. `http://localhost:3000/stats` → 연령·성별 매출, 카테고리, 시간대 통계 확인 (자동 갱신)

---

## 문제 해결

| 증상 | 원인 / 해결 |
|------|------------|
| `Failed to open camera` | 카메라 꺼짐/주소 불일치 → IP Webcam 실행 후 `--camera-url` 확인 |
| 영상은 나오는데 박스 없음 (`invalid_detections`) | 얼굴이 작거나 흐림/저조도 → 얼굴을 크고 또렷하게. 정상 폐기 동작 |
| `Low quality face - skipped` | 흐림/저조도/저대비 크롭 폐기 → 조명·초점 확인 |
| POS 결제했는데 연령 `알 수 없음` | 결제 시점에 얼굴 미인식(최근 2초 규칙) → 얼굴 비추며 결제 |
| 통계가 안 변함 | 로컬 백엔드(`backend_server.py`) 꺼짐 → 다시 실행 |
| `Using device: cpu` (느림) | GPU 미인식 → CUDA/PyTorch 설치 확인 |

RunPod(모드 B) 관련 문제는 [`실행가이드.md`](./실행가이드.md) 6장 참고.

---

## 모델 파일 & Git LFS

대용량 가중치는 일반 git이 아니라 다음 규칙으로 관리됩니다.

- **`ensemble_korean.pt`** (~786MB) — **Git LFS**로 추적합니다. clone 후 반드시:
  ```powershell
  git lfs install && git lfs pull
  ```
- **`ensemble_best.pt`** (~1.9GB), **`deploy_bundle.tar`** (~871MB), **`runpodctl.exe`** — 저장소에서 제외(`.gitignore`). 필요 시 별도 경로로 공유합니다.
- `fair_face_models/`의 가중치는 공개 FairFace pretrained 모델입니다.
- `.venv/`, `node_modules/`, `.next/`, SQLite `*.db`, `debug_crops/` 등은 커밋하지 않습니다.

> GitHub는 일반 파일 100MB 제한이 있어, LFS로 추적되지 않는 대용량 파일을 커밋하면 push가 실패합니다.

---

## 참고

- 성별 모델은 FairFace 기반 pretrained 모델을 사용합니다.
- 얼굴이 너무 작거나 품질이 낮으면 추론 결과를 의도적으로 폐기합니다.
- IP 카메라 주소는 실제 장치가 노출하는 스트림 URL과 일치해야 합니다.
- CORS 허용 origin은 환경변수 `ALLOWED_ORIGINS`(콤마 구분)로 제한할 수 있습니다(기본값: 전체 허용).
