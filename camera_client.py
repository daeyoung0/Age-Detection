"""Local camera client — runs on your PC.

Captures the IP/USB camera, shows the live window at full frame rate, and
sends frames to the RunPod GPU inference server asynchronously (so the video
never stalls while waiting for a prediction). Draws the latest result and
forwards it to the local backend for POS matching.

Example:
    python camera_client.py ^
        --camera-url http://192.168.219.101:8050/video ^
        --infer-url https://<POD_ID>-8000.proxy.runpod.net/infer ^
        --backend-url http://localhost:8000

Press 'q' to quit.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from collections import deque

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Local camera client for remote GPU inference")
    p.add_argument("--camera-url", type=str, default="", help="IP camera stream URL (empty = local webcam)")
    p.add_argument("--camera-index", type=int, default=0, help="Local camera index when --camera-url is empty")
    p.add_argument("--infer-url", type=str, required=True, help="RunPod inference endpoint, e.g. https://<id>-8000.proxy.runpod.net/infer")
    p.add_argument("--backend-url", type=str, default="http://localhost:8000", help="Local backend base URL")
    p.add_argument("--infer-interval", type=float, default=0.5, help="Seconds between inference requests to the GPU server")
    p.add_argument("--infer-timeout", type=float, default=25.0, help="Per-request timeout (s). Must exceed cold-start inference (~20s) to avoid a retry pile-up")
    p.add_argument("--send-width", type=int, default=0, help="Downscale frames to this width before sending (0 = full size, keeps faces large enough to pass detection). Lower = less bandwidth but smaller faces may be rejected")
    p.add_argument("--post-interval", type=float, default=1.0, help="Seconds between POSTs to the local backend")
    p.add_argument("--smooth", type=int, default=5, help="Average server probs over this many recent results (1 = off)")
    p.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for frames sent to the server")
    p.add_argument("--result-ttl", type=float, default=1.5, help="Drop the overlay if no fresh result within this many seconds")
    p.add_argument("--fullscreen", action="store_true", help="Start the camera window in fullscreen (toggle with 'f', quit with 'q')")
    return p.parse_args()


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None          # latest BGR frame for inference
        self.result = None         # latest parsed server result
        self.result_time = 0.0
        self.online = False
        self.running = True

    def set_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def set_result(self, result, online):
        with self.lock:
            self.result = result
            self.result_time = time.monotonic()
            self.online = online

    def get_result(self):
        with self.lock:
            return self.result, self.result_time, self.online


def post_backend(backend_url: str, age_group: str, face_size: int, gender: str):
    def _worker():
        try:
            body = json.dumps({"age_group": age_group, "face_size": face_size, "gender": gender}).encode()
            req = urllib.request.Request(
                f"{backend_url}/api/camera", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=2):
                pass
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


def infer_worker(state: SharedState, args):
    """Background thread: send latest frame to GPU server, store result."""
    smooth_hist = deque(maxlen=max(1, args.smooth))
    last_infer = 0.0
    last_post = 0.0

    while state.running:
        now = time.monotonic()
        if now - last_infer < args.infer_interval:
            time.sleep(0.01)
            continue
        frame = state.get_frame()
        if frame is None:
            time.sleep(0.02)
            continue
        last_infer = now

        # Downscale before sending to cut upload bandwidth over the tunnel.
        # Track the scale so the returned bbox can be mapped back to full res.
        scale = 1.0
        send = frame
        if args.send_width and frame.shape[1] > args.send_width:
            scale = args.send_width / frame.shape[1]
            send = cv2.resize(frame, (args.send_width, int(frame.shape[0] * scale)))

        ok, buf = cv2.imencode(".jpg", send, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            continue
        try:
            req = urllib.request.Request(
                args.infer_url, data=buf.tobytes(),
                headers={"Content-Type": "image/jpeg"}, method="POST",
            )
            # Timeout must exceed a cold-start inference (~20s) or the client
            # abandons + retries, piling requests onto the server until it stalls.
            with urllib.request.urlopen(req, timeout=args.infer_timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001
            state.set_result(state.get_result()[0], online=False)
            print(f"[client] infer request failed: {exc}")
            continue

        if data.get("status") != "ok":
            state.set_result({"status": data.get("status", "unknown")}, online=True)
            continue

        # map bbox back to full-resolution display coords
        if scale != 1.0 and data.get("bbox"):
            data["bbox"] = [int(v / scale) for v in data["bbox"]]

        # client-side temporal smoothing over the server's per-class probs
        probs = data.get("probs")
        if probs and args.smooth > 1:
            names = list(probs.keys())
            smooth_hist.append([probs[n] for n in names])
            mean = [sum(v[i] for v in smooth_hist) / len(smooth_hist) for i in range(len(names))]
            si = max(range(len(names)), key=lambda i: mean[i])
            data["age_group"] = names[si]
            data["confidence"] = mean[si]

        state.set_result(data, online=True)

        if now - last_post >= args.post_interval:
            post_backend(args.backend_url, data["age_group"], data.get("face_size", 0), data.get("gender", "unknown"))
            last_post = now


def open_capture(args):
    if args.camera_url:
        cap = cv2.VideoCapture(args.camera_url)
        name = args.camera_url
    else:
        cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(args.camera_index)
        name = f"index={args.camera_index}"
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {name}")
    # Minimize internal buffering so we don't fall behind real time (MJPEG lag).
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap, name


def capture_worker(state: "SharedState", cap):
    """Continuously drain the stream, keeping only the newest frame.

    IP-camera (MJPEG) streams buffer frames; if the display thread reads at its
    own pace the video falls progressively behind real time and looks laggy.
    Draining here means get_frame() always returns the latest frame.
    """
    while state.running:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        state.set_frame(frame)


def draw_overlay(frame, result, result_time, online, ttl):
    # connection indicator
    color = (0, 200, 0) if online else (0, 0, 255)
    cv2.putText(frame, f"GPU: {'online' if online else 'offline'}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if result is None or result.get("status") != "ok":
        status = "no result" if result is None else result.get("status", "unknown")
        cv2.putText(frame, f"Status: {status}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return
    if time.monotonic() - result_time > ttl:
        return  # stale, don't draw a box that no longer matches the scene

    x1, y1, x2, y2 = result["bbox"]
    gender = result.get("gender", "unknown")
    label = f"{result['age_group']} / {gender} ({result.get('confidence', 0):.2f})"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    cap, name = open_capture(args)
    print(f"[client] camera: {name}")
    print(f"[client] infer server: {args.infer_url}")
    print(f"[client] backend: {args.backend_url} | infer every {args.infer_interval}s | post every {args.post_interval}s")
    print("[client] press 'q' to quit")

    state = SharedState()
    grabber = threading.Thread(target=capture_worker, args=(state, cap), daemon=True)
    grabber.start()
    worker = threading.Thread(target=infer_worker, args=(state, args), daemon=True)
    worker.start()

    # Resizable window so Windows Snap (Win+Left/Right) and manual drag-resize work.
    window = "Camera Client (remote GPU inference) - q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            frame = state.get_frame()
            if frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            result, result_time, online = state.get_result()
            draw_overlay(frame, result, result_time, online, args.result_ttl)

            cv2.imshow(window, frame)
            # ~30 FPS display cap: avoids spinning the loop (and copying full
            # frames) hundreds of times/sec, which starves the capture thread.
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):  # toggle fullscreen on the fly
                full = cv2.getWindowProperty(window, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if full == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
    finally:
        state.running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
