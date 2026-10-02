import cv2
import mediapipe as mp
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VALID_SIGNS = ["Hello", "Yes", "No"]

TARGET_SAMPLES = 300

MODEL_PATH = "hand_landmarker.task"

LABEL = input("Enter sign name: ").strip().capitalize()

while LABEL not in VALID_SIGNS:
    print("Invalid sign name. Please enter a valid sign name.")
    LABEL = input("Enter sign name: ").strip().capitalize()

SAMPLES = 200

CSV_FILE = "dataset.csv"


base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(
    options
)


file_exists = os.path.exists(CSV_FILE)

file = open(
    CSV_FILE,
    "a",
    newline=""
)

writer = csv.writer(file)


if not file_exists:

    header = []

    for i in range(1, 22):
        header.append("x" + str(i))
        header.append("y" + str(i))
        header.append("z" + str(i))

    header.append("label")

    writer.writerow(header)


cap = cv2.VideoCapture(0)

count = 0


while True:

    success, frame = cap.read()

    if not success:
        print("Could not access webcam.")
        break


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    result = detector.detect(mp_image)


    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        data = []


        for landmark in hand_landmarks:

            data.append(landmark.x)
            data.append(landmark.y)
            data.append(landmark.z)


        data.append(LABEL)


        if count < SAMPLES:

            writer.writerow(data)

            count += 1


        for landmark in hand_landmarks:

            x = int(
                landmark.x * frame.shape[1]
            )

            y = int(
                landmark.y * frame.shape[0]
            )

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


    cv2.putText(
        frame,
        "Sign: " + LABEL,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Samples: " + str(count) + "/" + str(SAMPLES),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.imshow(
        "Dataset Collection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    if count >= SAMPLES:
        break


file.close()

cap.release()

cv2.destroyAllWindows()

detector.close()

print("Dataset collection completed.")