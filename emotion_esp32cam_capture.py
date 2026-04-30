import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import warnings
import urllib.request
from collections import deque

import cv2
import numpy as np

warnings.filterwarnings("ignore")


class HiddenPrints:
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._original_stdout = sys.stdout
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = self._original_stderr
        sys.stdout = self._original_stdout


with HiddenPrints():
    from deepface import DeepFace


ESP32_CAPTURE_URL = "http://172.20.10.5/capture"

EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "disgust", "fear"]

DISPLAY_NAMES = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "surprise": "surprised",
    "neutral": "neutral",
    "disgust": "disgust",
    "fear": "fear"
}


def average_scores(history):
    scores = {emotion: 0.0 for emotion in EMOTIONS}

    if not history:
        return scores

    for item in history:
        for emotion in EMOTIONS:
            scores[emotion] += item["scores"].get(emotion, 0.0)

    count = len(history)

    for emotion in EMOTIONS:
        scores[emotion] /= count

    return scores


def get_top_two(scores):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0], sorted_scores[0][1], sorted_scores[1][0], sorted_scores[1][1]


def get_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def get_snapshot():
    response = urllib.request.urlopen(ESP32_CAPTURE_URL, timeout=10)
    image_data = response.read()

    frame = cv2.imdecode(
        np.frombuffer(image_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    return frame


def main():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Error: Could not load face detector.")
        return

    print("ESP32-CAM snapshot emotion reader started.")
    print("Using:", ESP32_CAPTURE_URL)
    print("Press q to quit.")

    history = deque()
    window_seconds = 8.0

    stable_emotion = "neutral"
    stable_confidence = 0.0

    candidate_emotion = None
    candidate_count = 0
    needed_wins = 2

    min_best_score = 12.0
    min_gap = 4.0

    capture_interval = 2.0
    last_capture_time = 0

    last_frame = None
    last_face_status = "Waiting for snapshot..."

    while True:
        now = time.time()

        if now - last_capture_time >= capture_interval:
            last_capture_time = now

            try:
                frame = get_snapshot()

                if frame is None:
                    print("Snapshot failed: frame is empty.")
                    continue

                last_frame = frame.copy()
                display_frame = frame.copy()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20)
                )

                face = get_largest_face(faces)

                if face is not None:
                    last_face_status = "Face: detected"

                    x, y, w, h = face

                    cv2.rectangle(
                        display_frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        1
                    )

                    pad_left = int(0.12 * w)
                    pad_right = int(0.12 * w)
                    pad_top = int(0.22 * h)
                    pad_bottom = int(0.18 * h)

                    x1 = max(0, x - pad_left)
                    y1 = max(0, y - pad_top)
                    x2 = min(frame.shape[1], x + w + pad_right)
                    y2 = min(frame.shape[0], y + h + pad_bottom)

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        try:
                            small_face = cv2.resize(face_crop, (200, 200))

                            result = DeepFace.analyze(
                                img_path=small_face,
                                actions=["emotion"],
                                enforce_detection=False,
                                detector_backend="opencv",
                                align=False
                            )

                            if isinstance(result, list):
                                result = result[0]

                            history.append({
                                "time": now,
                                "scores": result["emotion"]
                            })

                        except Exception as error:
                            print("DeepFace skipped this frame:", error)

                else:
                    last_face_status = "Face: not detected"
                    display_frame = frame.copy()

                last_frame = display_frame

            except Exception as error:
                print("Snapshot error:", error)
                time.sleep(1)

        while history and (now - history[0]["time"] > window_seconds):
            history.popleft()

        scores = average_scores(history)
        best_emotion, best_score, second_emotion, second_score = get_top_two(scores)

        if {best_emotion, second_emotion} == {"angry", "disgust"}:
            if abs(best_score - second_score) < 5.0:
                best_emotion = stable_emotion

        if best_score >= min_best_score and (best_score - second_score) >= min_gap:
            if best_emotion == stable_emotion:
                candidate_emotion = None
                candidate_count = 0
            else:
                if candidate_emotion == best_emotion:
                    candidate_count += 1
                else:
                    candidate_emotion = best_emotion
                    candidate_count = 1

                if candidate_count >= needed_wins:
                    stable_emotion = candidate_emotion
                    candidate_emotion = None
                    candidate_count = 0

        stable_confidence = scores.get(stable_emotion, 0.0)

        if last_frame is not None:
            shown_name = DISPLAY_NAMES.get(stable_emotion, stable_emotion)
            runner_up = DISPLAY_NAMES.get(second_emotion, second_emotion)

            display_big = cv2.resize(last_frame, (480, 360))

            label1 = f"Emotion: {shown_name}"
            label2 = f"Confidence: {stable_confidence:.1f}%"
            label3 = f"Runner-up: {runner_up} ({second_score:.1f}%)"

            cv2.putText(display_big, label1, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(display_big, label2, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(display_big, label3, (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(display_big, last_face_status, (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(display_big, "Snapshot mode: updates every 2 sec", (20, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("ESP32-CAM Snapshot Emotion Reader", display_big)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
