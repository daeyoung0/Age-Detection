# 연령대 분석 시스템 (Age-Detection)

실시간 카메라 영상에서 얼굴을 검출하고 **연령대 6분류**를 추론하여, POS(결제) 데이터와 매칭해 연령대별 구매 통계를 내는 시스템입니다.

- **카메라 추론**: RetinaFace로 얼굴 검출 → 한국인 데이터로 파인튜닝한 ConvNeXt-Large 모델로 연령대 분류
- **백엔드**: FastAPI + SQLite. 카메라 결과 저장, 결제 시점 연령대 매칭, 통계 제공
- **프론트**: `index.html` (대시보드)

---

## 연령대 클래스

| 클래스 | 나이 범위 | 표시 라벨(백엔드) |
|--------|-----------|-------------------|
| `infant` | 0–5 | 유아/아동 (0-12) |
| `child` | 6–12 | 유아/아동 (0-12) |
| `teen` | 13–18 | 청소년 (13-18) |
| `young_adult` | 19–34 | 청년층 (19-34) |
| `middle_aged` | 35–49 | 중장년층 (35-49) |
| `senior` | 50+ | 고령층 (50+) |

---

## 모델

기본 모델은 **`ensemble_korean.pt`** 입니다.

- **아키텍처**: ConvNeXt-Large (timm), 입력 224×224
- **학습**: AIHub 안면 에이징 데이터(한국인)로 파인튜닝 (`finetune_korean_swin.py`)
- **포맷**: 카메라 로더(`load_ensemble_from_checkpoint`)가 읽을 수 있도록 단일 모델을 앙상블 번들 형식(모델 1개, weight `[1.0]`)으로 감쌌습니다. 즉 **이름에 `ensemble`이 있지만 실제로는 한국 파인튜닝 단독 모델**입니다.

### 성능 (한국인 val, subject 분리 · 데이터 누수 없음, `eval_korean.py`)

- **전체 정확도 ≈ 82.7–83.3%**, macro-F1 ≈ **0.83**

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|----|
| infant | 0.95 | 0.84 | 0.89 |
| child | 0.85 | 0.92 | 0.89 |
| teen | 0.77 | 0.80 | 0.79 |
| young_adult | 0.82 | 0.85 | 0.83 |
| middle_aged | 0.80 | 0.71 | 0.75 |
| senior | 0.81 | 0.81 | 0.81 |

> 주 혼동은 인접 경계(teen↔young_adult, middle_aged↔young_adult/senior). `middle_aged`가 가장 약함.

### 왜 한국 단독 모델인가
기존 `ensemble_best.pt`(3-모델 앙상블)는 학습 데이터에서 **인종과 나이가 엮여**(어린 클래스=한국인, 성인 클래스=서양인 MORPH) 있어, 한국인 성인을 teen/child로 오인식했습니다. 한국인 성인을 학습한 파인튜닝 모델로 교체해 이 문제를 해결했습니다. (`ensemble_best.pt`도 폴더에 남아있어 `--checkpoint ensemble_best.pt`로 비교 가능)

---

## 설치

Python 3.10+ (개발 환경 3.11), CUDA 12.1 권장.

```powershell
# GPU (권장)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
# 나머지 의존성
pip install -r requirements.txt
```

`timm`은 학습 시점 버전(`0.8.13dev0` 등 0.8.x dev)과의 호환을 위해 `model.py`에 버전 톨러런트 로더가 포함돼 있습니다.

---

## 실행

### 1) 백엔드 서버

```powershell
uvicorn backend_server:app --host 0.0.0.0 --port 8000
```

- `POST /api/camera` — 카메라가 보내는 `{age_group, face_size}` 저장
- `POST /api/pay` — 결제 직전 2초 내 마지막 연령대를 구매에 매칭
- `GET /api/stats` — 연령대별 건수/매출 통계
- DB는 기본적으로 OS 임시폴더의 `age-detection-main/store_data.db` 에 저장 (환경변수 `AGE_DETECTION_DB_PATH`로 변경 가능)

### 2) 카메라 추론

```powershell
# 로컬 웹캠
python camera_infer_swin.py --disable-tta

# IP 카메라 (휴대폰 등) — 주소 형식 주의: http://IP:PORT/경로
python camera_infer_swin.py --camera-url http://192.168.219.101:8050/video --disable-tta
```

`q` 키로 종료. 초록 박스 위에 `연령대 (신뢰도)` 가 표시되고, 1초마다 백엔드로 POST됩니다(`SENT` 표시).

---

## 카메라 추론 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--camera-url` | (없음) | IP 카메라 스트림 URL. 비우면 로컬 웹캠 사용 |
| `--camera-index` | `0` | 로컬 카메라 인덱스 |
| `--checkpoint` | `ensemble_korean.pt` | 사용할 모델 번들 |
| `--disable-tta` | off | TTA 끔. **권장** — TTA의 center-crop이 얼굴을 더 조여 어리게 볼 수 있고, 정확도 수치도 TTA 없이 측정됨 |
| `--face-margin` | `0.15` | 검출 박스 확장 비율. **학습 크롭(0.15)과 일치**시킨 값 |
| `--min-score` | `0.8` | RetinaFace 검출 점수 최소값 (약한 오검출 제거) |
| `--min-face` | `80` | 최소 얼굴 크기(px). 너무 작은 박스 무시 |
| `--smooth` | `8` | 최근 N프레임 확률 평균(temporal smoothing). `1`=끔 |
| `--clahe` | off | CLAHE 대비 보정. **학습/추론 불일치**라 기본 꺼짐, A/B 테스트용 |
| `--debug` | off | 6클래스 확률 전체 출력 + `debug_crops/`에 실제 입력 크롭 저장 |
| `--post-interval` | `1.0` | 백엔드 POST 간격(초) |
| `--max-size` | `1024` | RetinaFace 입력 최대 크기 |

---

## 추론 파이프라인 (전처리)

학습과 **동일한** 전처리를 사용합니다 (이게 가장 중요):

1. 프레임 BGR → **RGB** 변환
2. RetinaFace로 얼굴 검출, 가장 큰 **유효** 얼굴 선택
   - 검출점수 < `--min-score`, `--min-face` 미만, 가로/세로 > 1.5(띠 모양) 박스는 **버림**
3. 박스를 `--face-margin`(0.15)만큼 확장해 크롭 → 학습 크롭 프레이밍과 일치
4. `Resize((224,224))` + ImageNet 정규화(mean `0.485,0.456,0.406` / std `0.229,0.224,0.225`)
5. 모델 추론 → softmax 확률 → 최근 `--smooth`프레임 평균 → argmax

라이브러리 진입점은 `pipeline_swin.py` (단일 이미지/배치용):
```python
from pipeline_swin import load_ensemble
pipeline = load_ensemble("ensemble_korean.pt")
result = pipeline.predict("image.jpg")
```

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|------|-------------|
| `Failed to open camera source` | IP 주소 형식 확인 — 포트 앞은 `:` (점 아님), `http://IP:PORT/video`. 폰·PC 동일 와이파이. 브라우저에서 그 주소가 열리는지 확인 |
| 50대가 **child/infant**로 튐 | 대부분 **찌그러진/작은 검출 박스**(움직임·MJPEG 깨짐)에서 발생. 디테일이 뭉개지면 모델이 "매끈함=젊음"으로 붕괴. `--min-score`/`--min-face` 필터 + `--smooth` 평균으로 억제됨. 정상 크롭에선 senior로 잘 나옴 |
| 청년이 teen으로 | teen↔young_adult 경계 혼동(모델 한계). margin 0.15 + `--disable-tta`로 완화 |
| 라벨이 매 프레임 바뀜 | `--smooth` 값을 키움(예: `12`) |
| 저조도에서 부정확 | 조명 개선이 우선. `--clahe`로 실험 가능하나 학습 불일치라 실측으로 효과 확인 후 사용 |
| 스트림 자주 끊김(`Stream ends prematurely`) | IP 카메라 네트워크 불안정(모델 무관). 와이파이/앱 재시작 |
| `FileNotFoundError: ensemble_korean.pt` | 프로젝트 폴더 안에서 실행 (`cd` 후 실행) |

---

## 디버깅 (모델이 실제로 보는 것 확인)

```powershell
python camera_infer_swin.py --camera-url <URL> --disable-tta --debug
```
- 콘솔: `[debug] bbox=WxHpx | smoothed=... | raw <6클래스 확률>`
- `debug_crops/` 폴더에 모델 입력 크롭 저장 → 검출이 얼굴을 제대로 잡았는지 눈으로 확인

---

## 파일 구조

```
Age-Detection-main/
├── camera_infer_swin.py     # 실시간 카메라 추론 (메인 실행)
├── pipeline_swin.py         # 단일 이미지 추론 라이브러리 API
├── backend_server.py        # FastAPI 서버 (카메라 저장 / 결제 매칭 / 통계)
├── index.html               # 대시보드 프론트
├── ensemble_korean.pt       # 기본 모델 (한국 파인튜닝 ConvNeXt-Large, ~750MB)
├── ensemble_best.pt         # 구 3-모델 앙상블 (비교용, ~1.9GB)
├── src/swinface_age/        # 모델/전처리 코드 (model.py, dataset.py)
└── requirements.txt
```

> 학습/평가 스크립트(`finetune_korean_swin.py`, `eval_korean.py` 등)와 데이터셋은 별도 학습 프로젝트(`E:\cap_swin`)에 있습니다.
