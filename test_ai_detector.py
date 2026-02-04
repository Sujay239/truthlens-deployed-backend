from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

def test_model(model_name):
    print(f"\nTesting Model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Test Case 1: Human News (Real) - Simple, factual
        real_text = "The city council met yesterday to discuss the new budget proposal. The meeting lasted three hours."
        
        # Test Case 2: AI Generated News (Real-looking but AI) - polished, generic
        ai_text = "In a groundbreaking development, scientists have discovered a method to synthesize infinite energy using quantum crystals found in the ocean floor."
        
        test_cases = [
            (real_text, "HUMAN/REAL"),
            (ai_text, "AI/FAKE")
        ]
        
        for text, expected in test_cases:
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
            p0 = probs[0][0].item()
            p1 = probs[0][1].item()
            
            print(f"Text: '{text[:30]}...' (Exp: {expected})")
            print(f"  Label 0: {p0:.4f}")
            print(f"  Label 1: {p1:.4f}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # OpenAI Detector (Label 0=Real, Label 1=Fake/AI) usually
    test_model("roberta-base-openai-detector")
