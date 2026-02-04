
import os
import pandas as pd
import shutil
import logging
from video_detector import DeepfakeVideoDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_VIDEO_DIR = os.path.join(BASE_DIR, "data for video")
CSV_PATH = os.path.join(DATA_VIDEO_DIR, "DeepFake Videos Dataset.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_frames")

def prepare_dataset():
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    
    os.makedirs(os.path.join(PROCESSED_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "fake"), exist_ok=True)

    try:
        df = pd.read_csv(CSV_PATH)
        logger.info(f"Loaded CSV with {len(df)} records.")
        
        detector = DeepfakeVideoDetector()
        
        for index, row in df.iterrows():
            # CSV columns: id, deepfake, image, video
            # deepfake -> path like "deepfake/1.mp4" (Fake label)
            # video -> path like "video/1.mp4" (Real label)
            
            fake_rel_path = row['deepfake']
            real_rel_path = row['video']
            
            fake_full_path = os.path.join(DATA_VIDEO_DIR, fake_rel_path)
            real_full_path = os.path.join(DATA_VIDEO_DIR, real_rel_path)
            
            # Extract frames for Fake
            if os.path.exists(fake_full_path):
                # We create a sub-folder per video to avoid name collisions, or just frame_X_video_Y
                # Simplified: just dump all frames into 'fake' with unique prefix
                prefix = f"vid_{index}_fake"
                # Temporary logic: extract to a temp folder then move/rename
                temp_out = os.path.join(PROCESSED_DIR, "temp")
                count = detector.extract_frames(fake_full_path, temp_out, max_frames=20)
                
                for f in os.listdir(temp_out):
                    shutil.move(os.path.join(temp_out, f), os.path.join(PROCESSED_DIR, "fake", f"{prefix}_{f}"))
                os.rmdir(temp_out)
                logger.info(f"Extracted {count} frames from {fake_rel_path}")
            else:
                logger.warning(f"File not found: {fake_full_path}")

            # Extract frames for Real
            if os.path.exists(real_full_path):
                prefix = f"vid_{index}_real"
                temp_out = os.path.join(PROCESSED_DIR, "temp")
                count = detector.extract_frames(real_full_path, temp_out, max_frames=20)
                
                for f in os.listdir(temp_out):
                    shutil.move(os.path.join(temp_out, f), os.path.join(PROCESSED_DIR, "real", f"{prefix}_{f}"))
                os.rmdir(temp_out)
                logger.info(f"Extracted {count} frames from {real_rel_path}")
            else:
                logger.warning(f"File not found: {real_full_path}")
                
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")

def main():
    logger.info("Step 1: Preparing dataset...")
    prepare_dataset()
    
    logger.info("Step 2: Training model...")
    detector = DeepfakeVideoDetector()
    detector.train_model(PROCESSED_DIR, epochs=3)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
