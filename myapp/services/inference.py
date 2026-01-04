import cv2
import os
import time
import uuid
import threading
from pathlib import Path
from typing import Generator, Tuple, List
import numpy as np
import pytesseract
from ultralytics import YOLO
from django.conf import settings
from myapp.models import Violation

YOLO_HELMET_WEIGHTS = os.getenv("YOLO_HELMET_WEIGHTS", "yolov8n.pt")
YOLO_PLATE_WEIGHTS = os.getenv("YOLO_PLATE_WEIGHTS", "yolov8n.pt")

helmet_model = YOLO(YOLO_HELMET_WEIGHTS)
plate_model = YOLO(YOLO_PLATE_WEIGHTS)

def _annotate_frame(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = det["label"]
        conf = det["conf"]
        color = (0, 200, 0) if det.get("ok", False) else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame

def _run_plate_ocr(plate_crop: np.ndarray) -> str:
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = r"--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    return "".join([c for c in text if c.isalnum()]).upper()

def _detect_plates(frame) -> List[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    plates = []
    results = plate_model.predict(source=frame, imgsz=640, verbose=False)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            plates.append((crop, (x1, y1, x2, y2)))
    return plates

def process_video_to_mjpeg(file_path: str, session_id: str) -> Generator[bytes, None, None]:
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    delay = 1.0 / fps
    snapshot_dir = Path(settings.MEDIA_ROOT) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = helmet_model.predict(source=frame, imgsz=640, verbose=False)
        detections = []
        overload_flag = False
        no_helmet_flag = False

        for r in results:
            for b in r.boxes:
                cls_name = helmet_model.names[int(b.cls)]
                conf = float(b.conf)
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                det = {"bbox": (x1, y1, x2, y2), "label": cls_name, "conf": conf}
                detections.append(det)

        persons = [d for d in detections if "person" in d["label"]]
        if len(persons) > 2:
            overload_flag = True
        if any("no_helmet" in d["label"] for d in detections):
            no_helmet_flag = True

        violations = []
        if overload_flag:
            violations.append(("overload", max([p["conf"] for p in persons], default=0.5)))
        if no_helmet_flag:
            nh_conf = max([d["conf"] for d in detections if "no_helmet" in d["label"]], default=0.5)
            violations.append(("no_helmet", nh_conf))

        plate_text = ""
        if violations:
            plates = _detect_plates(frame)
            if plates:
                crop, bbox = plates[0]
                plate_text = _run_plate_ocr(crop)

        for vtype, vconf in violations:
            snap_name = f"{session_id}_{uuid.uuid4().hex}.jpg"
            snap_path = snapshot_dir / snap_name
            cv2.imwrite(str(snap_path), frame)
            Violation.objects.create(
                violation_type=vtype,
                plate_text=plate_text,
                confidence=vconf,
                snapshot=f"snapshots/{snap_name}",
                meta={"session": session_id},
            )

        annotated = _annotate_frame(frame.copy(), detections)
        ret2, jpeg = cv2.imencode(".jpg", annotated)
        if not ret2:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(delay)

    cap.release()