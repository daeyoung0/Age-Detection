# RunPod GPU 추론 배포 가이드

무거운 추론(RetinaFace + 연령 앙상블 + FairFace)만 RunPod GPU에서 돌리고,
폰 카메라 캡처와 화면 표시는 **내 PC**에서 합니다.

```
[폰 IP카메라] → [내 PC: camera_client.py]  ──프레임(JPEG)──▶  [RunPod: infer_server.py (GPU)]
                      │  ◀──결과(age/gender)────────────────────────┘
                      ├─▶ 화면(cv2 창)에 박스 그리기
                      └─▶ 로컬 백엔드(/api/camera)로 결과 전송 → POS/통계 연동
```

---

## 1. RunPod 쪽 (GPU 추론 서버)

### 1-1. Pod 생성
- 템플릿: **RunPod PyTorch 2.x** (CUDA 포함) 계열
- GPU: 아무 거나 가능 (RTX 3090/4090, A4000 등이면 충분)
- **HTTP Ports**에 `8000` 추가 (노출 필요)

### 1-2. 코드 + 모델 업로드
Git LFS로 올리는 게 깔끔합니다 (`.gitattributes`에 `*.pt`가 LFS로 잡혀 있음):

```bash
# Pod 터미널에서
git lfs install
git clone <이 저장소 URL> app && cd app
git lfs pull                     # ensemble_korean.pt(786MB) 등 실제 파일 받기
```

Git을 안 쓰면 `runpodctl send` 또는 Pod의 Jupyter 업로드로
다음을 올립니다: `infer_server.py`, `camera_infer_swin.py`, `src/`,
`ensemble_korean.pt`, `fair_face_models/`, `requirements.txt`.

### 1-3. 의존성 설치
PyTorch 템플릿엔 torch가 이미 있으니, 나머지만:

```bash
pip install timm retinaface-pytorch insightface onnxruntime \
            opencv-python-headless pillow numpy fastapi uvicorn pydantic
```
> RunPod은 headless라 `opencv-python` 대신 **`opencv-python-headless`** 권장.
> `facenet-pytorch`는 이 체크포인트엔 불필요(설치하지 말 것).

### 1-4. 서버 실행
```bash
python -m uvicorn infer_server:app --host 0.0.0.0 --port 8000
```
로그에 `[infer] device: cuda` 가 떠야 GPU 사용 중입니다.
처음엔 RetinaFace 가중치(96MB)를 자동 다운로드합니다.

### 1-5. 외부 접속 URL 확인
RunPod이 노출 포트마다 프록시 URL을 만듭니다:
```
https://<POD_ID>-8000.proxy.runpod.net
```
헬스 체크:
```
https://<POD_ID>-8000.proxy.runpod.net/health
→ {"device":"cuda","cuda":true,...}
```

---

## 2. 내 PC 쪽 (카메라 + 화면 + 백엔드/프론트)

### 2-1. 백엔드 + 프론트 (기존과 동일)
```powershell
# 터미널 1 - 백엔드
.\.venv\Scripts\python.exe -m uvicorn backend_server:app --host 0.0.0.0 --port 8000

# 터미널 2 - 프론트
npm run dev
```

### 2-2. 카메라 클라이언트 (RunPod에 추론 요청)
```powershell
# 터미널 3
.\.venv\Scripts\python.exe camera_client.py `
  --camera-url http://192.168.219.101:8050/video `
  --infer-url https://<POD_ID>-8000.proxy.runpod.net/infer `
  --backend-url http://localhost:8000
```
- 영상은 풀 FPS로 부드럽게 표시되고, 추론은 0.3초마다 비동기로 RunPod에 요청합니다.
- 좌상단 `GPU: online/offline` 로 서버 연결 상태 확인.
- 종료: 영상 창에서 `q`.

### 옵션
| 플래그 | 기본 | 설명 |
|--------|------|------|
| `--infer-interval` | 0.3 | RunPod 추론 요청 간격(초). GPU 빠르면 0.1로 |
| `--post-interval` | 1.0 | 백엔드 전송 간격(초) |
| `--smooth` | 5 | 최근 N개 결과 평균(라벨 안정화) |
| `--jpeg-quality` | 80 | 전송 프레임 화질(낮추면 대역폭↓) |
| `--result-ttl` | 1.5 | 이 시간 내 새 결과 없으면 박스 숨김 |

---

## 3. 참고
- 포트 충돌 주의: RunPod의 `infer_server`도 8000, 로컬 백엔드도 8000을 쓰지만
  서로 다른 머신이라 무관합니다. 클라이언트는 둘을 `--infer-url`(RunPod)과
  `--backend-url`(로컬)로 구분해서 호출합니다.
- 폰(IP Webcam)이 자주 잠들면 앱에서 화면 꺼짐 방지/배터리 최적화 예외를 켜세요.
- 보안: RunPod 프록시 URL은 공개 주소이므로, 민감하면 `infer_server`에 간단한
  토큰 헤더 검사를 추가하는 걸 권장(필요하면 붙여드림).
```
