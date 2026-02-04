import time
print("Staritng checks...")

start = time.time()
print("Importing torch...")
import torch
print(f"Torch imported in {time.time() - start:.2f}s")

start = time.time()
print("Importing transformers...")
from transformers import BertTokenizer, BertForSequenceClassification
print(f"Transformers imported in {time.time() - start:.2f}s")

print("Imports successful!")
