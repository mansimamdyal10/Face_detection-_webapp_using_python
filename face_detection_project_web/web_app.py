from flask import Flask, render_template, request
from pathlib import Path
import uuid
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = Path("static/uploads")
RESULT_FOLDER = Path("static/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# -------------------------------
#  OpenCV Face Detection Function
# -------------------------------
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    return faces

# -------------------------------
# Load, Save, Annotate Functions
# -------------------------------
def load_image(path):
    return cv2.imread(str(path))

def save_image(img, path):
    cv2.imwrite(str(path), img)

def annotate_image(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


# -------------------------------
# Flask Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Save uploaded file
        file_ext = Path(file.filename).suffix
        filename = f"{uuid.uuid4().hex}{file_ext}"
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)

        # Process image
        img = load_image(upload_path)
        faces = detect_faces(img)
        annotated = annotate_image(img, faces)

        result_filename = f"{uuid.uuid4().hex}{file_ext}"
        result_path = RESULT_FOLDER / result_filename
        save_image(annotated, result_path)

        return render_template(
            "index.html",
            uploaded_image=f"/{upload_path.as_posix()}",
            result_image=f"/{result_path.as_posix()}",
            face_count=len(faces)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
