from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import uuid

from face_detection.detector import detect_faces
from face_detection.utils import load_image, save_image, annotate_image

app = Flask(__name__)
UPLOAD_FOLDER = Path("static/uploads")
RESULT_FOLDER = Path("static/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file part")
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        # Save upload
        file_ext = Path(file.filename).suffix
        filename = f"{uuid.uuid4().hex}{file_ext}"
        upload_path = UPLOAD_FOLDER / filename
        file.save(upload_path)

        # Process
        img = load_image(upload_path)
        faces = detect_faces(img)
        annotated = annotate_image(img, faces)

        result_filename = f"{uuid.uuid4().hex}{file_ext}"
        result_path = RESULT_FOLDER / result_filename
        save_image(result_path, annotated)

        return render_template("index.html",
                               uploaded_image=f"/{upload_path.as_posix()}",
                               result_image=f"/{result_path.as_posix()}",
                               face_count=len(faces))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
