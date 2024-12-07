import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from pyntcloud import PyntCloud
from transformers import BertTokenizer, BertModel

# --- Параметры ---
latent_dim = 128  # Размерность латентного пространства
text_embedding_dim = 768  # Размерность текстового эмбеддинга BERT
use_text_embeddings = True  # Использовать ли текстовые эмбеддинги
epochs = 10
batch_size = 32
learning_rate = 1e-3
num_points = 1024  # Количество точек для выборки из облака точек

# --- Загрузка предобученной модели BERT ---
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
# --- Функции для обработки данных и текста ---
def preprocess_text(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    return torch.tensor([input_ids])

def get_text_embedding(text):
    with torch.no_grad():
        outputs = bert_model(text)
        embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

# --- Класс датасета ---
class ModelNetDataset(Dataset):
    def __init__(self, root_dir, split='train', categories=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.categories = categories
        self.transform = transform

        self.data = []
        self.labels = []

        for category in os.listdir(root_dir):
            if categories is not None and category not in categories:
                continue

            category_dir = os.path.join(root_dir, category, split)
            for filename in os.listdir(category_dir):
                if filename.endswith('.off'):
                    self.data.append(os.path.join(category_dir, filename))
                    self.labels.append(category)

    def __len__(self):
        return len(self.data)

