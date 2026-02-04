import sys
import os

# Ensure the app module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app.ml.train_model import train_model
except ImportError as e:
    print(f"Error importing train_model: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("="*50)
    print("STARTING FULL MODEL TRAINING")
    print("="*50)
    print("This process will fine-tune BERT on your full dataset.")
    print("It may take several hours on a CPU.")
    print("Using 512 Token Length (Maximum capacity).")
    print("Defaulting to 5 Epochs.")
    print("="*50)
    
    try:
        train_model(epochs=5)
        print("\n" + "="*50)
        print("TRAINING SUCCESSFUL!")
        print("Required model file 'bert_fake_news.pth' has been created.")
        print("You can now restart your server and usage the Fake News Detection.")
        print("="*50)
    except Exception as e:
        print(f"\nTraining Failed: {e}")
