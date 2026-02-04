import torch
import torch.nn.functional as F
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from .constants import MODEL_SAVE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Base class for managing model lifecycle"""
    def __init__(self, name):
        self.name = name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        raise NotImplementedError

class FakeNewsModel(ModelManager):
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
             cls._instance = FakeNewsModel("FakeNews")
        return cls._instance

    def load(self):
        if self.model and self.tokenizer:
            return

        logger.info("Loading Fake News Model...")
        if os.path.exists(MODEL_SAVE_PATH):
            logger.info(f"Loading custom Fake News model from {MODEL_SAVE_PATH}")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            self.model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=self.device))
        else:
            logger.info("No custom Fake News model found. Loading specialized pre-trained model: mrm8488/bert-tiny-finetuned-fake-news-detection")
            # Using a fine-tuned model for better accuracy out-of-the-box
            model_name = "mrm8488/bert-tiny-finetuned-fake-news-detection"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("Fake News Model loaded.")

    def predict(self, text):
        self.load()
        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
                
                # Assume Index 0 = Real, Index 1 = Fake (Standard for fake news binary)
                real_prob = probs[0][0].item()
                fake_prob = probs[0][1].item()
                
                label = "Fake" if fake_prob > real_prob else "Real"
                confidence = fake_prob if label == "Fake" else real_prob
                
            return {
                "label": label,
                "confidence": confidence,
                "fake_probability": fake_prob,
                "real_probability": real_prob
            }
        except Exception as e:
            logger.error(f"Fake News Prediction Failed: {e}")
            return {"label": "Error", "confidence": 0.0, "error": str(e)}

class AiTextModel(ModelManager):
    _instance = None
    # Switched from 'roberta-base-openai-detector' to a more modern ChatGPT-aware model
    MODEL_NAME = 'Hello-SimpleAI/chatgpt-detector-roberta'

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
             cls._instance = AiTextModel("AiText")
        return cls._instance

    def load(self):
        if self.model and self.tokenizer:
            return

        logger.info(f"Loading AI Text Model ({self.MODEL_NAME})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info("AI Text Model loaded.")
        except Exception as e:
            logger.error(f"Failed to load AI Text Model: {e}")

    def predict(self, text):
        self.load()
        if not self.model:
            return {"label": "Error", "error": "Model failed to load"}

        try:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False, # RoBERTa doesn't use token_type_ids
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs.logits, dim=1)
                
                # For roberta-base-openai-detector: 0=Fake(AI), 1=Real(Human)
                ai_prob = probs[0][0].item()
                human_prob = probs[0][1].item()
                
                is_ai = ai_prob > human_prob
                label = "AI Generated" if is_ai else "Human Written"
                confidence = ai_prob if is_ai else human_prob
                
            return {
                "label": label,
                "confidence": confidence,
                "is_ai": is_ai,
                "ai_probability": ai_prob,
                "human_probability": human_prob
            }
        except Exception as e:
            logger.error(f"AI Text Prediction Failed: {e}")
            return {"label": "Error", "error": str(e)}


# -- Public API --

def predict_fake_news(text):
    return FakeNewsModel.get_instance().predict(text)

def predict_ai_generated_text(text):
    return AiTextModel.get_instance().predict(text)

def get_model_and_tokenizer():
    # Legacy support if needed, but we should deprecate
    # Returning the Fake News model components by default to avoid breaking things that expect it
    instance = FakeNewsModel.get_instance()
    instance.load()
    return instance.model, instance.tokenizer

