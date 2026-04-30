import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import json
import joblib
import warnings
from pathlib import Path

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

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


BASE_DIR = Path("images")
OUTPUT_DIR = Path("knn_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURE_CSV = OUTPUT_DIR / "emotion_features.csv"
ARRAY_JSON = OUTPUT_DIR / "emotion_arrays.json"
MODEL_FILE = OUTPUT_DIR / "emotion_knn_model.pkl"
SCATTER_FILE = OUTPUT_DIR / "emotion_scatter_plot.png"
RESULTS_FILE = OUTPUT_DIR / "knn_results.txt"

EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral", "disgust", "fear"]

FEATURE_COLUMNS = [
    "happy_score",
    "sad_score",
    "angry_score",
    "surprise_score",
    "neutral_score",
    "disgust_score",
    "fear_score"
]

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda rect: rect[2] * rect[3])


def crop_face_like_esp32(frame, face):
    x, y, w, h = face

    pad_left = int(0.12 * w)
    pad_right = int(0.12 * w)
    pad_top = int(0.22 * h)
    pad_bottom = int(0.18 * h)

    x1 = max(0, x - pad_left)
    y1 = max(0, y - pad_top)
    x2 = min(frame.shape[1], x + w + pad_right)
    y2 = min(frame.shape[0], y + h + pad_bottom)

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    return cv2.resize(face_crop, (200, 200))


def extract_deepface_scores(image_path, face_cascade):
    frame = cv2.imread(str(image_path))

    if frame is None:
        return None, "failed", "image_read_failed"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20)
    )

    face = get_largest_face(faces)

    if face is None:
        return None, "failed", "no_face_detected"

    face_crop = crop_face_like_esp32(frame, face)

    if face_crop is None:
        return None, "failed", "bad_face_crop"

    try:
        result = DeepFace.analyze(
            img_path=face_crop,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            align=False
        )

        if isinstance(result, list):
            result = result[0]

        scores = result["emotion"]
        deepface_prediction = result["dominant_emotion"]

        feature_array = [
            float(scores.get("happy", 0.0)),
            float(scores.get("sad", 0.0)),
            float(scores.get("angry", 0.0)),
            float(scores.get("surprise", 0.0)),
            float(scores.get("neutral", 0.0)),
            float(scores.get("disgust", 0.0)),
            float(scores.get("fear", 0.0))
        ]

        return feature_array, deepface_prediction, "success"

    except Exception as error:
        return None, "failed", f"deepface_error: {error}"


def build_feature_dataset():
    print("\n==============================")
    print("STEP 1: EXTRACTING DEEPFACE FEATURES")
    print("==============================")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        print("Error: Could not load face detector.")
        return None

    rows = []
    emotion_arrays = {emotion: [] for emotion in EMOTIONS}

    for true_emotion in EMOTIONS:
        folder = BASE_DIR / true_emotion

        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        image_files = [
            file for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
        ]

        print(f"\nProcessing {true_emotion}: {len(image_files)} images")

        for index, image_path in enumerate(image_files, start=1):
            feature_array, deepface_prediction, status = extract_deepface_scores(
                image_path,
                face_cascade
            )

            row = {
                "filename": image_path.name,
                "path": str(image_path),
                "true_emotion": true_emotion,
                "deepface_prediction": deepface_prediction,
                "status": status
            }

            if feature_array is not None:
                emotion_arrays[true_emotion].append(feature_array)

                for column, value in zip(FEATURE_COLUMNS, feature_array):
                    row[column] = value
            else:
                for column in FEATURE_COLUMNS:
                    row[column] = ""

            rows.append(row)

            if index % 25 == 0:
                print(f"  {index}/{len(image_files)} done")

    df = pd.DataFrame(rows)
    df.to_csv(FEATURE_CSV, index=False)

    with open(ARRAY_JSON, "w") as file:
        json.dump(emotion_arrays, file, indent=4)

    print("\nSaved DeepFace feature data to:", FEATURE_CSV)
    print("Saved organized emotion arrays to:", ARRAY_JSON)

    print("\nArray counts:")
    for emotion in EMOTIONS:
        print(f"{emotion}_array: {len(emotion_arrays[emotion])}")

    return df


def evaluate_deepface(df):
    print("\n==============================")
    print("STEP 2: DEEPFACE-ONLY BASELINE")
    print("==============================")

    usable = df[df["status"] == "success"].copy()

    y_true = usable["true_emotion"]
    y_pred = usable["deepface_prediction"]

    accuracy = accuracy_score(y_true, y_pred)

    print(f"DeepFace-only accuracy: {accuracy * 100:.2f}%")

    print("\nDeepFace classification report:")
    print(classification_report(y_true, y_pred, labels=EMOTIONS, zero_division=0))

    return accuracy


def train_knn(df):
    print("\n==============================")
    print("STEP 3: TRAINING KNN MODEL")
    print("==============================")

    usable = df[df["status"] == "success"].copy()

    for column in FEATURE_COLUMNS:
        usable[column] = pd.to_numeric(usable[column], errors="coerce")

    usable = usable.dropna(subset=FEATURE_COLUMNS)

    X = usable[FEATURE_COLUMNS]
    y = usable["true_emotion"]

    print("Total usable samples:", len(usable))
    print("\nSamples by emotion:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    best_k = None
    best_accuracy = 0
    best_model = None
    best_predictions = None

    k_values = [1, 3, 5, 7, 9, 11, 13]

    print("\nTesting K values:")

    for k in k_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=k,
                weights="distance"
            ))
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f"K = {k}: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
            best_model = model
            best_predictions = predictions

    report = classification_report(
        y_test,
        best_predictions,
        labels=EMOTIONS,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        best_predictions,
        labels=EMOTIONS
    )

    print("\n==============================")
    print("KNN FINAL RESULTS")
    print("==============================")
    print(f"Best K: {best_k}")
    print(f"KNN accuracy: {best_accuracy * 100:.2f}%")

    print("\nKNN classification report:")
    print(report)

    print("\nKNN confusion matrix:")
    print("Labels:", EMOTIONS)
    print(matrix)

    model_bundle = {
        "model": best_model,
        "feature_columns": FEATURE_COLUMNS,
        "emotions": EMOTIONS,
        "best_k": best_k
    }

    joblib.dump(model_bundle, MODEL_FILE)

    with open(RESULTS_FILE, "w") as file:
        file.write("KNN FINAL RESULTS\n")
        file.write("==============================\n")
        file.write(f"Best K: {best_k}\n")
        file.write(f"KNN accuracy: {best_accuracy * 100:.2f}%\n\n")
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write("Labels: " + str(EMOTIONS) + "\n")
        file.write(str(matrix))

    print("\nSaved KNN model to:", MODEL_FILE)
    print("Saved KNN results to:", RESULTS_FILE)

    return best_accuracy


def create_scatter_plot(df):
    print("\n==============================")
    print("STEP 4: CREATING SCATTER PLOT")
    print("==============================")

    usable = df[df["status"] == "success"].copy()

    for column in FEATURE_COLUMNS:
        usable[column] = pd.to_numeric(usable[column], errors="coerce")

    usable = usable.dropna(subset=FEATURE_COLUMNS)

    X = usable[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    usable["pca_x"] = X_2d[:, 0]
    usable["pca_y"] = X_2d[:, 1]

    plt.figure(figsize=(10, 7))

    for emotion in EMOTIONS:
        subset = usable[usable["true_emotion"] == emotion]

        if len(subset) == 0:
            continue

        plt.scatter(
            subset["pca_x"],
            subset["pca_y"],
            label=emotion,
            alpha=0.7
        )

    plt.title("DeepFace Emotion Score Arrays Mapped with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(SCATTER_FILE, dpi=300)
    plt.close()

    print("Saved scatter plot to:", SCATTER_FILE)


def main():
    print("Starting local DeepFace + KNN emotion training pipeline...")

    df = build_feature_dataset()

    if df is None:
        return

    deepface_accuracy = evaluate_deepface(df)
    knn_accuracy = train_knn(df)
    create_scatter_plot(df)

    print("\n==============================")
    print("ALL DONE")
    print("==============================")
    print(f"DeepFace-only accuracy: {deepface_accuracy * 100:.2f}%")
    print(f"DeepFace + KNN accuracy: {knn_accuracy * 100:.2f}%")

    print("\nFiles created:")
    print(FEATURE_CSV)
    print(ARRAY_JSON)
    print(MODEL_FILE)
    print(SCATTER_FILE)
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()
