
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import io
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "resnet_deepfake.pth")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data for image", "cropped_images")

class FFDataset(Dataset):
    """
    Custom Dataset for FaceForensics++ structure.
    Folders with '_' in name are usually manipulated (000_003).
    Folders with single number (000) are original.
    Fallback: If no original found, we might need manual labeling or assume structure.
    Here we implement logic: 
    - Contains '_': Label 1 (Fake)
    - No '_': Label 0 (Real)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        # Walk through all subdirectories
        if not os.path.exists(self.root_dir):
            return

        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            # Determine label
            # FF++ convention: 000_003 (Fake), 000 (Real)
            if "_" in folder_name:
                label = 1 # Fake
            else:
                label = 0 # Real
            
            # Get all images in folder
            images = glob.glob(os.path.join(folder_path, "*.png")) + \
                     glob.glob(os.path.join(folder_path, "*.jpg"))
            
            for img_path in images:
                self.image_paths.append(img_path)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a generic zero tensor fallback or skip (simplified here)
            return torch.zeros(3, 224, 224), label

class DeepfakeImageDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeImageDetector, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Initializes the ResNet model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing DeepfakeImageDetector on {self.device}...")

        try:
            # Load ResNet18 (Standard CNN)
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            
            # Modify the final fully connected layer for Binary Classification (Real vs Fake)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)

            self.model = self.model.to(self.device)
            
            # Transforms
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Check for existing weights
            if os.path.exists(MODEL_WEIGHTS_PATH):
                logger.info(f"Loading custom model weights from {MODEL_WEIGHTS_PATH}")
                self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=self.device))
                self.model.eval()
            else:
                logger.warning("No custom weights found.")
                # AUTO-TRAINING LOGIC
                if os.path.exists(DATA_PATH):
                    logger.info(f"Data directory found at {DATA_PATH}. Starting auto-training...")
                    self.train_model()
                else:
                    logger.warning(f"Data directory not found at {DATA_PATH}. Skipping training.")
                    self.model.eval() # Fallback to ImageNet weights

            logger.info("DeepfakeImageDetector initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize DeepfakeImageDetector: {e}")
            self.model = None

    def train_model(self, epochs=1): # Doing 1 epoch for quick startup, normally more
        """Trains the model on the available data."""
        try:
            dataset = FFDataset(DATA_PATH, transform=self.transform)
            
            if len(dataset) == 0:
                logger.warning("Dataset is empty. Skipping training.")
                return

            # Check class balance
            real_count = dataset.labels.count(0)
            fake_count = dataset.labels.count(1)
            logger.info(f"Training Data: {len(dataset)} images. Real: {real_count}, Fake: {fake_count}")
            
            if real_count == 0 or fake_count == 0:
                 logger.warning("Dataset classes are unbalanced (needs both Real and Fake). Skipping training to prevent bias.")
                 return

            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            
            self.model.train()
            logger.info("Starting training loop...")
            
            for epoch in range(epochs):
                running_loss = 0.0
                for i, data in enumerate(dataloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    
                logger.info(f"Epoch {epoch+1}/{epochs} finished. Loss: {running_loss/len(dataloader):.4f}")

            # Save weights
            torch.save(self.model.state_dict(), MODEL_WEIGHTS_PATH)
            logger.info(f"Model trained and saved to {MODEL_WEIGHTS_PATH}")
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.model.eval()

    def predict(self, image_bytes):
        """
        Predicts whether an image is Real or Fake.
        """
        if self.model is None:
            return {"label": "Error", "confidence": 0.0, "error": "Model not initialized"}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Index 0 = Real, Index 1 = Fake
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()

                if fake_prob > real_prob:
                    label = "Fake"
                    confidence = fake_prob
                else:
                    label = "Real"
                    confidence = real_prob

                return {
                    "label": label,
                    "confidence": confidence,
                    "fake_probability": fake_prob,
                    "real_probability": real_prob
                }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"label": "Error", "confidence": 0.0, "error": str(e)}

# Global Accessor
def predict_image(image_bytes):
    detector = DeepfakeImageDetector()
    return detector.predict(image_bytes)
