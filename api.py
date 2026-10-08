# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------
# CREATE FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="ISL Sign Recognition API",
    description="AI-based Indian Sign Language recognition",
    version="1.0"
)


# ---------------------------------------------------------
# ALLOW FRONTEND TO COMMUNICATE WITH API
# ---------------------------------------------------------

# This allows the browser frontend to send images
# to our FastAPI server.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------------------------------------------------
# LOAD TRAINED ML MODEL
# ---------------------------------------------------------

# sign_model.pkl contains our trained Random Forest model.

with open("sign_model.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------------
# LOAD MEDIAPIPE HAND LANDMARK MODEL
# ---------------------------------------------------------

# This file detects the 21 landmarks of a hand.

MODEL_PATH = "hand_landmarker.task"


# Tell MediaPipe where the model file is located.

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


# Configure the hand detector.
# Our current ML model was trained using one hand.

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)


# Create the MediaPipe detector.

detector = vision.HandLandmarker.create_from_options(
    options
)


# ---------------------------------------------------------
# CONFIDENCE THRESHOLD
# ---------------------------------------------------------

# If the model is less than 70% confident,
# we don't force it to choose Hello, Yes or No.

CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.get("/")
def home():

    # Send our HTML frontend to the browser.

    return FileResponse("index.html")


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health():

    # This lets us quickly check whether the server
    # is running correctly.

    return {
        "status": "running",
        "model": "loaded",
        "message": "ISL Sign Recognition API is working"
    }


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # -----------------------------------------------------
    # 1. Read the image sent by the browser
    # -----------------------------------------------------

    image_bytes = await file.read()


    # -----------------------------------------------------
    # 2. Convert image bytes into NumPy data
    # -----------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # -----------------------------------------------------
    # 3. Convert NumPy data into an OpenCV image
    # -----------------------------------------------------

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # If the image could not be decoded,
    # return an error instead of crashing.

    if frame is None:

        return {
            "prediction": "No Sign",
            "confidence": 0,
            "error": "Invalid image"
        }


    # -----------------------------------------------------
    # 4. Convert BGR to RGB
    # -----------------------------------------------------

    # OpenCV uses BGR.
    # MediaPipe expects RGB.

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # 5. Create MediaPipe image
    # -----------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------------------------------
    # 6. Detect hand landmarks
    # -----------------------------------------------------

    result = detector.detect(mp_image)


    # If no hand is detected, return No Sign.

    if not result.hand_landmarks:

        return {
            "prediction": "No Sign",
            "confidence": 0
        }


    # -----------------------------------------------------
    # 7. Get the first detected hand
    # -----------------------------------------------------

    hand_landmarks = result.hand_landmarks[0]


    # -----------------------------------------------------
    # 8. Extract 63 landmark features
    # -----------------------------------------------------

    # There are 21 hand landmarks.
    #
    # Each landmark contains:
    # X
    # Y
    # Z
    #
    # 21 × 3 = 63 features.

    data = []


    for landmark in hand_landmarks:

        data.append(landmark.x)
        data.append(landmark.y)
        data.append(landmark.z)


    # -----------------------------------------------------
    # 9. Create the feature names
    # -----------------------------------------------------

    # IMPORTANT:
    #
    # Your training data uses:
    #
    # x1, y1, z1
    # x2, y2, z2
    # ...
    # x21, y21, z21
    #
    # So prediction must use exactly the same names.

    feature_names = []


    for i in range(1, 22):

        feature_names.append(f"x{i}")
        feature_names.append(f"y{i}")
        feature_names.append(f"z{i}")


    # -----------------------------------------------------
    # 10. Create DataFrame for the ML model
    # -----------------------------------------------------

    # Using a DataFrame keeps the feature names consistent
    # with the data used during model training.

    features = pd.DataFrame(
        [data],
        columns=feature_names
    )


    # -----------------------------------------------------
    # 11. Get prediction probabilities
    # -----------------------------------------------------

    # The Random Forest gives us the probability of
    # each known sign.

    probabilities = model.predict_proba(
        features
    )[0]


    # -----------------------------------------------------
    # 12. Find the strongest prediction
    # -----------------------------------------------------

    best_index = probabilities.argmax()


    # Get the predicted sign.

    predicted_class = model.classes_[best_index]


    # Get the confidence of that prediction.

    confidence = float(
        probabilities[best_index]
    )


    # -----------------------------------------------------
    # 13. Apply confidence threshold
    # -----------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        prediction = predicted_class

    else:

        prediction = "No Sign"


    # -----------------------------------------------------
    # 14. Send result to frontend
    # -----------------------------------------------------

    return {
        "prediction": str(prediction),
        "confidence": round(confidence, 3)
    }