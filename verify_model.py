from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def test_model(model_name):
    print(f"\nTesting Model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        test_cases = [
            ("Aliens have landed in New York and are eating pizza.", "FAKE"),
            ("The stock market closed higher today amid positive economic data.", "REAL")
        ]
        
        for text, expected in test_cases:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
            p0 = probs[0][0].item()
            p1 = probs[0][1].item()
            
            print(f"Text: '{text[:20]}...' (Exp: {expected})")
            print(f"  P0: {p0:.4f}")
            print(f"  P1: {p1:.4f}")
            
            winner = 0 if p0 > p1 else 1
            print(f"  Winner: Index {winner}")
            
    except Exception as e:
        print(f"Error testing {model_name}: {e}")

if __name__ == "__main__":
    test_model("mrm8488/bert-tiny-finetuned-fake-news-detection")
