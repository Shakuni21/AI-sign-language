import cv2
import mediapipe as mp
import pickle
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------
# 1. Load the trained ML model
# ---------------------------------------------------------

# This is the MediaPipe hand-landmark model.
MODEL_PATH = "hand_landmarker.task"


# Load the Random Forest model that we trained earlier.
with open("sign_model.pkl", "rb") as file:
    model = pickle.load(file)


# ---------------------------------------------------------
# 2. Set up MediaPipe Hand Landmarker
# ---------------------------------------------------------

# Tell MediaPipe where the hand detection model is located.
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


# Configure the hand detector.
# We are currently using only one hand.
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)


# Create the hand detector.
detector = vision.HandLandmarker.create_from_options(
    options
)


# ---------------------------------------------------------
# 3. Open the webcam
# ---------------------------------------------------------

# 0 means the default webcam.
cap = cv2.VideoCapture(0)


# ---------------------------------------------------------
# 4. Confidence threshold
# ---------------------------------------------------------

# The prediction will only be accepted if
# the model is at least 70% confident.

CONFIDENCE_THRESHOLD = 0.70


# ---------------------------------------------------------
# 5. Start the webcam loop
# ---------------------------------------------------------

while True:

    # Read one frame from the webcam.
    success, frame = cap.read()


    # If the camera fails, stop the program.
    if not success:

        print("Could not access webcam.")
        break


    # -----------------------------------------------------
    # 6. Convert BGR to RGB
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
    # 7. Detect the hand
    # -----------------------------------------------------

    result = detector.detect(mp_image)


    # Default prediction.
    prediction = "No Sign"


    # Default confidence.
    confidence = 0.0


    # -----------------------------------------------------
    # 8. Check whether a hand was detected
    # -----------------------------------------------------

    if result.hand_landmarks:

        # Take the first detected hand.
        hand_landmarks = result.hand_landmarks[0]


        # This list will contain 63 features.
        data = []


        # -------------------------------------------------
        # 9. Extract x, y and z from all 21 landmarks
        # -------------------------------------------------

        for landmark in hand_landmarks:

            # Add X coordinate.
            data.append(landmark.x)

            # Add Y coordinate.
            data.append(landmark.y)

            # Add Z coordinate.
            data.append(landmark.z)


        # -------------------------------------------------
        # 10. Create the feature names
        # -------------------------------------------------

        # IMPORTANT:
        #
        # Our training dataset uses:
        #
        # x1, y1, z1
        # x2, y2, z2
        # ...
        # x21, y21, z21
        #
        # Therefore we MUST use the same names here.

        feature_names = []


        for i in range(1, 22):

            feature_names.append(f"x{i}")
            feature_names.append(f"y{i}")
            feature_names.append(f"z{i}")


        # -------------------------------------------------
        # 11. Convert data into a DataFrame
        # -------------------------------------------------

        # This makes the prediction data have the
        # exact same feature names as the training data.

        features_df = pd.DataFrame(
            [data],
            columns=feature_names
        )


        # -------------------------------------------------
        # 12. Get probabilities from the ML model
        # -------------------------------------------------

        # The model gives us the probability for
        # Hello, Yes and No.

        probabilities = model.predict_proba(
            features_df
        )[0]


        # Find the position of the highest probability.
        best_index = probabilities.argmax()


        # Get the highest probability.
        confidence = probabilities[best_index]


        # Get the name of the predicted class.
        predicted_class = model.classes_[best_index]


        # -------------------------------------------------
        # 13. Apply confidence threshold
        # -------------------------------------------------

        # Only accept the prediction when the model
        # is sufficiently confident.

        if confidence >= CONFIDENCE_THRESHOLD:

            prediction = predicted_class

        else:

            prediction = "No Sign"


        # -------------------------------------------------
        # 14. Draw hand landmarks
        # -------------------------------------------------

        for landmark in hand_landmarks:

            # Convert normalized coordinates into
            # actual webcam pixel coordinates.

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )


            # Draw the landmark.
            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


    # -----------------------------------------------------
    # 15. Display prediction
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "Prediction: " + prediction,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # -----------------------------------------------------
    # 16. Display confidence
    # -----------------------------------------------------

    cv2.putText(
        frame,
        "Confidence: " +
        str(round(confidence * 100, 1)) +
        "%",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # -----------------------------------------------------
    # 17. Show webcam
    # -----------------------------------------------------

    cv2.imshow(
        "ISL Sign Recognition",
        frame
    )


    # -----------------------------------------------------
    # 18. Press Q to quit
    # -----------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ---------------------------------------------------------
# 19. Release everything
# ---------------------------------------------------------

cap.release()

cv2.destroyAllWindows()

detector.close()