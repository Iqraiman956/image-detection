from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os

app = FastAPI()

# Serve frontend website
app.mount("/website", StaticFiles(directory="static", html=True), name="website")

# Serve detected images
app.mount("/results", StaticFiles(directory="."), name="results")

# Load YOLO model
model = YOLO("yolo11n.pt")


@app.get("/")
def home():
    return {"message": "YOLO Object Detection API is running!"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # Save uploaded image
    input_path = "uploaded_image.jpg"

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Run YOLO detection
    results = model(input_path, conf=0.35)

    # Create image with bounding boxes
    result = results[0]
    result.save(filename="detected_image.jpg")

    # Collect detected objects
    detections = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        detections.append({
            "object": model.names[class_id],
            "confidence": round(confidence * 100, 1)
        })

    return {
        "image_url": "/results/detected_image.jpg",
        "detections": detections
    }