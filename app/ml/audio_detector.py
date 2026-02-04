import os
import joblib
import numpy as np
import librosa
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "audio_classifier.joblib")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "audio_label_encoder.joblib")

class AudioDeepfakeDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioDeepfakeDetector, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        self.model = None
        self.encoder = None
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.encoder = joblib.load(ENCODER_PATH)
                logger.info("Audio model and encoder loaded successfully.")
            else:
                logger.warning("Audio model not found. Please run train_audio_model.py.")
        except Exception as e:
            logger.error(f"Failed to load audio model: {e}")

    def extract_features(self, audio_path):
        """
        Extracts the exact same features as the training dataset (DATASET-balanced.csv).
        Features: chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfcc1...mfcc20
        """
        try:
            y, sr = librosa.load(audio_path, duration=30) # Load up to 30s
            
            # If audio is too short, pad it? Or loop it? 
            # Ideally we extract features from 1-second chunks and average, or just one chunk.
            # The dataset says "features extracted from each 1 second of audio".
            # For simplicity, we calculate global stats (mean) which mimics the 1s aggregations if the clip is short.
            # Or better, we take the mean of the features over the time axis.
            
            chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # mfcc is (20, T). We take mean across time to get 20 feature values.
            mfccs = np.mean(mfcc, axis=1) # shape (20,)
            
            # Construct feature vector in correct order
            # Order: chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfcc1...mfcc20
            features = [chroma_stft, rms, spec_cent, spec_bw, rolloff, zcr]
            features.extend(mfccs)
            
            return np.array(features).reshape(1, -1) # (1, 26)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def predict(self, audio_path):
        if self.model is None:
             return {"label": "Error", "confidence": 0.0, "error": "Model not trained"}

        features = self.extract_features(audio_path)
        if features is None:
            return {"label": "Error", "confidence": 0.0, "error": "Could not extract features"}

        try:
            probs = self.model.predict_proba(features)[0] # [prob_class0, prob_class1]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            label = self.encoder.inverse_transform([pred_idx])[0]
            
            return {
                "label": label, # REAL or FAKE (based on dataset)
                "confidence": confidence,
                "is_fake": label.upper() == "FAKE" 
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"label": "Error", "confidence": 0.0, "error": str(e)}

# Global Accessor
def predict_audio(audio_path):
    detector = AudioDeepfakeDetector()
    return detector.predict(audio_path)
