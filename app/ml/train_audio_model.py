import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data for voice")
CSV_PATH = os.path.join(DATA_DIR, "DATASET-balanced.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_classifier.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "audio_label_encoder.joblib")

def train_model():
    print(f"Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: Dataset not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Feature columns (based on metadata/csv observation)
    # The dataset typically has 'chroma_stft', 'rms', 'spectral_centroid', ..., 'mfcc1'...'mfcc20', 'LABEL'
    # We drop the label to get features
    X = df.drop('LABEL', axis=1)
    y = df['LABEL']
    
    # Encode labels (REAL/FAKE)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save encoder
    joblib.dump(le, ENCODER_PATH)
    print(f"Label encoder saved to {ENCODER_PATH}")
    print(f"Classes: {le.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
