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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ---------------------------------------------------------
# LOAD OUR TRAINED ML MODEL
# ---------------------------------------------------------

# sign_model.pkl contains the Random Forest model
# that we trained using Hello, Yes and No data.

with open("sign_model.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------------
# LOAD MEDIAPIPE HAND LANDMARK MODEL
# ---------------------------------------------------------

# This is the MediaPipe model we already used
# in our previous hand detection program.

MODEL_PATH = "hand_landmarker.task"


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


# Our current ML dataset uses one hand,
# so we detect one hand here.

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

# If the model is less confident than this value,
# we will return "No Sign" instead of forcing
# Hello, Yes or No.

CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------
# FRONTEND PAGE
# ---------------------------------------------------------

@app.get("/")
def home():

    # Send index.html when someone opens the website.

    return FileResponse("index.html")


# ---------------------------------------------------------
# PREDICTION API
# ---------------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read the image sent by the browser.

    image_bytes = await file.read()


    # Convert image bytes into a NumPy array.

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )


    # Convert the NumPy array into an OpenCV image.

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )


    # Make sure the image was read correctly.

    if frame is None:

        return {
            "prediction": "No Sign",
            "confidence": 0,
            "error": "Invalid image"
        }


    # -----------------------------------------------------
    # CONVERT BGR TO RGB
    # -----------------------------------------------------

    # OpenCV uses BGR.
    # MediaPipe expects RGB.

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Convert the image into a MediaPipe Image.

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------------------------------
    # DETECT HAND
    # -----------------------------------------------------

    result = detector.detect(mp_image)


    # If MediaPipe cannot find a hand,
    # there is nothing to classify.

    if not result.hand_landmarks:

        return {
            "prediction": "No Sign",
            "confidence": 0
        }


    # Get the first detected hand.

    hand_landmarks = result.hand_landmarks[0]


    # -----------------------------------------------------
    # EXTRACT 63 FEATURES
    # -----------------------------------------------------

    # Our model was trained using:
    #
    # 21 landmarks
    # ×
    # 3 coordinates (x, y, z)
    #
    # = 63 features

    data = []


    for landmark in hand_landmarks:

        data.append(landmark.x)
        data.append(landmark.y)
        data.append(landmark.z)


    # Convert the list into the format
    # expected by the ML model.

    features = np.array(data).reshape(1, -1)


    # -----------------------------------------------------
    # GET MODEL PROBABILITIES
    # -----------------------------------------------------

    # Instead of only asking for the prediction,
    # get the probability of every class.

    probabilities = model.predict_proba(features)[0]


    # Find the class with the highest probability.

    best_index = probabilities.argmax()


    # Get the name of that class.

    predicted_class = model.classes_[best_index]


    # Get its probability.

    confidence = float(
        probabilities[best_index]
    )


    # -----------------------------------------------------
    # APPLY CONFIDENCE THRESHOLD
    # -----------------------------------------------------

    if confidence >= CONFIDENCE_THRESHOLD:

        prediction = predicted_class

    else:

        prediction = "No Sign"


    # -----------------------------------------------------
    # SEND RESULT BACK TO BROWSER
    # -----------------------------------------------------

    return {
        "prediction": prediction,
        "confidence": round(confidence, 3)
    }