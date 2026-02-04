import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "news data for training")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "bert_fake_news.pth")
