import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------

# Read our collected hand-landmark data from the CSV file.
# The "label" column contains the sign name:
# Hello, Yes, No, etc.

df = pd.read_csv("dataset.csv")


# ---------------------------------------------------------
# 2. Separate features and labels
# ---------------------------------------------------------

# X contains the input features.
# These are the 63 hand-landmark values:
# x1, y1, z1 ... x21, y21, z21

X = df.drop("label", axis=1)


# y contains the correct sign for each sample.
# Example: Hello, Yes, No

y = df["label"]


# ---------------------------------------------------------
# 3. Split data into training and testing sets
# ---------------------------------------------------------

# 80% of the data will be used to train the model.
# 20% will be kept separate for testing.
#
# random_state=42 makes the split reproducible.
# stratify=y keeps the sign classes balanced between
# training and testing data.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. Create the Random Forest model
# ---------------------------------------------------------

# Random Forest combines many decision trees to make
# the final prediction.
#
# n_estimators=100 means the forest contains 100 trees.
# random_state=42 makes the model reproducible.

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ---------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------

# Give the training data to the Random Forest.
# The model learns the relationship between the
# hand landmarks (X_train) and sign names (y_train).

model.fit(X_train, y_train)


# ---------------------------------------------------------
# 6. Test the trained model
# ---------------------------------------------------------

# Use the test data, which the model did not use
# during training, to make predictions.

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 7. Calculate accuracy
# ---------------------------------------------------------

# Compare the model's predictions with the actual
# labels from the test dataset.

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)


# ---------------------------------------------------------
# 8. Generate a classification report
# ---------------------------------------------------------

# This gives us precision, recall and F1-score
# for each sign class.

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ---------------------------------------------------------
# 9. Save the trained model
# ---------------------------------------------------------

# Save the trained Random Forest so that we don't
# have to train it every time we run prediction.

with open("sign_model.pkl", "wb") as file:

    pickle.dump(
        model,
        file
    )


print("\nModel saved as sign_model.pkl")