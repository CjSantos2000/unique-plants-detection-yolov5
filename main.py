# import pathlib

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


import io
from flask import Flask, request, jsonify, abort
from PIL import Image
import torch


app = Flask(__name__)


@app.route("/")
def hello():
    return "Crop & Corn Detection API"


@app.route("/detect-unique-plants/", methods=["POST"])
def detect_uniq_image():
    if "image" not in request.files:
        abort(400, "No image part")

    image_file = request.files["image"]

    # Check if the file has a valid content type (e.g., image/jpeg, image/png, >
    if not allowed_file(image_file.filename):
        abort(415, "Unsupported Media Type")

    model = initialize_model()

    print(image_file)

    try:
        # Read the image data and save it to a file
        image_data = image_file.read()
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        result_dict = {
            "detected": "None",
            "confidence": "None",
            "message": "Invalid image format, Please upload a valid image",
        }
        return jsonify(result_dict), 400

    results = model(img, size=640)
    result = None
    detected = []

    try:
        detected.append(
            (results.pandas().xyxy[0].name[0], results.pandas().xyxy[0].confidence[0])
        )
    except:
        result_dict = {
            "detected": "None",
            "confidence": "None",
            "message": "No unique plant detected, Please Try Again",
        }
        return jsonify(result_dict), 200

    if detected:
        result = detected[0]

    result_dict = {
        "detected": result[0],
        "confidence": result[1],
        "message": "Successfully detected unique plant",
    }

    return jsonify(result_dict), 200


def initialize_model():
    model = torch.hub.load("yolov5", "custom", path="models/uniq.pt", source="local")
    # set model parameters
    model.conf = 0.5  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image

    return model


# Define a function to check if the file extension is allowed
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == "__main__":
    app.run()
