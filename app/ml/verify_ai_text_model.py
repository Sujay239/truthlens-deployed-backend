from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def verify_model():
    model_name = 'roberta-base-openai-detector'
    print(f"Loading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        texts = [
            "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system.",
            "I am an artificial intelligence model created by OpenAI."
        ]
        
        print("\nTesting inference...")
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = F.softmax(logits, dim=1)
                
            # label 0 is Fake/AI, label 1 is Real/Human (usually)
            print(f"Text: {text[:50]}...")
            print(f"Probs: {probs[0].tolist()}")
            
        print("\n✅ Model loaded and working.")
        
    except Exception as e:
        print(f"❌ Model failed: {e}")

if __name__ == "__main__":
    verify_model()
