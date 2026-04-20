import time
from pathlib import Path

import cv2
import requests
from ultralytics import YOLO

CHANNEL_ACCESS_TOKEN = "yvmzrSgTQySLP3WeEN09ff02LrVjDlGU4i7yHUBFKKq/DD2Mj7Ehs+6zte2dpPb+7/k7h+7WMDpSIJ00PgngXjf4MRgLVetP3EmG94D+jnxA24XBtX+JpKd+YMDwearyBEKUIvOUN7nFrMT//UsrPwdB04t89/1O/w1cDnyilFU="
TO_USER_ID = "U68643be81d1347c8fa43ef4706b378e3"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("錯誤：無法開啟攝影機")
    raise SystemExit

save_dir = Path(r"C:\Users\yc_ja\Desktop\captures")
save_dir.mkdir(exist_ok=True)

video_dir = save_dir / "videos"
video_dir.mkdir(exist_ok=True)

cooldown_seconds = 10
no_person_timeout = 10

last_save_time = 0.0
last_line_time = 0.0
last_seen_time = time.time()

recording = False
save_enabled = True
line_enabled = True

video_writer = None
video_path = None

window_name = "AI Monitor"

screen_width = 1920
screen_height = 1080

windowed_width = 960
windowed_height = 540

is_fullscreen = False

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, windowed_width, windowed_height)

def send_line_message(text):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": "Bearer " + CHANNEL_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    payload = {
        "to": TO_USER_ID,
        "messages": [
            {
                "type": "text",
                "text": text
            }
        ]
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        print("LINE 狀態碼：", r.status_code)
        print(r.text)
    except Exception as e:
        print("LINE 發送失敗：", repr(e))

def start_recording(frame):
    global video_writer, video_path
    if video_writer is not None:
        return

    filename = video_dir / f"video_{time.strftime('%Y%m%d_%H%M%S')}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(
        str(filename),
        fourcc,
        20.0,
        (frame.shape[1], frame.shape[0])
    )
    video_path = filename
    print(f"開始錄影：{video_path}")

def stop_recording():
    global video_writer, video_path
    if video_writer is not None:
        video_writer.release()
        print(f"停止錄影：{video_path}")
        video_writer = None
        video_path = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("錯誤：無法讀取攝影機畫面")
            break

        results = model(frame, conf=0.25, verbose=False)
        r = results[0]
        names = r.names

        person_detected = False
        person_count = 0

        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id].lower()

            if label != "person":
                continue

            person_detected = True
            person_count += 1
            last_seen_time = time.time()

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.0%}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2
            )

        now = time.time()

        if person_detected and save_enabled and now - last_save_time >= cooldown_seconds:
            filename = save_dir / f"person_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"已自動存圖：{filename}")
            last_save_time = now

            if line_enabled and now - last_line_time >= cooldown_seconds:
                send_line_message(f"Person detected: {filename.name}")
                last_line_time = now

        if recording:
            if video_writer is None:
                start_recording(frame)
            if video_writer is not None:
                video_writer.write(frame)
        else:
            stop_recording()

        if now - last_seen_time > no_person_timeout:
            print("畫面已一段時間沒有人，自動關閉")
            break

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (260, 170), (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        cv2.putText(
            frame,
            f"Persons: {person_count}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Rec: {'ON' if recording else 'OFF'}",
            (15, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1
        )

        cv2.putText(
            frame,
            f"Save: {'ON' if save_enabled else 'OFF'}",
            (15, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 0),
            1
        )

        cv2.putText(
            frame,
            f"LINE: {'ON' if line_enabled else 'OFF'}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 0, 255),
            1
        )

        cv2.putText(
            frame,
            f"[R] Rec  [S] Save",
            (15, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"[L] LINE [F] Full",
            (15, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        cv2.putText(
            frame,
            f"[Q] Quit",
            (15, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1
        )

        if is_fullscreen:
            display_frame = cv2.resize(frame, (screen_width, screen_height))
        else:
            display_frame = frame

        cv2.imshow(window_name, display_frame)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("視窗已關閉")
            break

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            recording = not recording
            print(f"錄影功能：{'開啟' if recording else '關閉'}")
        elif key == ord("s"):
            save_enabled = not save_enabled
            print(f"存圖功能：{'開啟' if save_enabled else '關閉'}")
        elif key == ord("l"):
            line_enabled = not line_enabled
            print(f"LINE 通知：{'開啟' if line_enabled else '關閉'}")
        elif key == ord("f"):
            is_fullscreen = not is_fullscreen

            if is_fullscreen:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
                print("全螢幕模式")
            else:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )
                cv2.resizeWindow(window_name, windowed_width, windowed_height)
                print("小視窗模式")

finally:
    stop_recording()
    cap.release()
    cv2.destroyAllWindows()