"""
Step 2 of the ML pipeline: Feature extraction + train classifiers.

This script:
1. Loads combined_dataset.csv (created by load_dataset.py)
2. Filters to only the clean, usable activity labels (empty, walking, falling)
3. Extracts features from each CSI reading
4. Splits into train/test sets
5. Trains two classifiers: SVM and Random Forest
6. Reports accuracy and a confusion matrix for each
7. Saves the best model + scaler + label encoder to disk for real-time inference

Run this after load_dataset.py:
    python train_classifier.py
"""

import pandas as pd
import numpy as np
import ast

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib


# ==== CONFIGURATION ====
INPUT_FILE = "combined_dataset.csv"
USABLE_LABELS = ["empty", "walking", "falling", "sitting_standing", "lying_down", "running", "bending", "one_person", "two_person"]
TEST_SIZE = 0.25
RANDOM_STATE = 42
# ========================


def extract_features(csi_array):
    arr = np.array(csi_array, dtype=float)
    stats = [
        np.mean(arr),
        np.std(arr),
        np.min(arr),
        np.max(arr),
        np.median(arr),
        np.sum(np.abs(arr)),
        np.mean(np.abs(np.diff(arr)))
    ]
    return np.concatenate([stats, arr])


def main():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df["csi_values"] = df["csi_values"].apply(ast.literal_eval)

    print(f"Total samples before filtering: {len(df)}")
    df = df[df["activity_label"].isin(USABLE_LABELS)].reset_index(drop=True)
    print(f"Total samples after filtering to {USABLE_LABELS}: {len(df)}")

    standard_len = df["csi_values"].apply(len).mode()[0]
    df = df[df["csi_values"].apply(len) == standard_len].reset_index(drop=True)
    print(f"Total samples after removing length outliers: {len(df)}")

    print("\nFinal class distribution:")
    print(df["activity_label"].value_counts())

    X = np.array([extract_features(row) for row in df["csi_values"]])
    y = df["activity_label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classes: {list(le.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

    print("\n" + "=" * 50)
    print("SVM Classifier")
    print("=" * 50)
    svm = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"Accuracy: {svm_acc:.2%}")
    print(confusion_matrix(y_test, svm_pred))
    print(classification_report(y_test, svm_pred, target_names=le.classes_, zero_division=0))

    print("\n" + "=" * 50)
    print("Random Forest Classifier")
    print("=" * 50)
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Accuracy: {rf_acc:.2%}")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred, target_names=le.classes_, zero_division=0))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"SVM Accuracy:           {svm_acc:.2%}")
    print(f"Random Forest Accuracy: {rf_acc:.2%}")

    best_model = rf if rf_acc >= svm_acc else svm
    best_name = "Random Forest" if rf_acc >= svm_acc else "SVM"
    print(f"\nBest model: {best_name}")

    joblib.dump(best_model, "trained_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print(f"\nSaved best model ({best_name}) to trained_model.pkl")
    print("Saved scaler.pkl and label_encoder.pkl")
    print("Ready for real-time inference.")


if __name__ == "__main__":
    main()