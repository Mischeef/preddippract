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

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        # Загружаем облако точек из файла .off
        pointcloud = PyntCloud.from_file(path)
        points = pointcloud.points.values

        # Применяем преобразования (опционально)
        if self.transform:
            points = self.transform(points)

        return points, label
    
    # --- Функция для выборки фиксированного количества точек ---
def sample_points(pointcloud, num_points):
    """Выбирает фиксированное количество точек из облака точек."""
    pointcloud = torch.tensor(pointcloud).float()
    N, C = pointcloud.shape
    if N < num_points:
        # Дублируем точки, если их меньше, чем нужно
        pointcloud = pointcloud.repeat(int(np.ceil(num_points / N)), 1)
    elif N > num_points:
        # Выбираем случайное подмножество точек
        idx = torch.randperm(N)[:num_points]
        pointcloud = pointcloud[idx]
    return pointcloud

# --- Класс VAE ---
class VAE(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim

     # --- Энкодер ---
        # Вход: облако точек (N, 3)
        self.enc_conv1 = nn.Conv1d(3, 64, 1)
        self.enc_conv2 = nn.Conv1d(64, 128, 1)
        self.enc_conv3 = nn.Conv1d(128, 256, 1)
        self.enc_fc_mu = nn.Linear(256 + text_embedding_dim, latent_dim)
        self.enc_fc_logvar = nn.Linear(256 + text_embedding_dim, latent_dim)

         # --- Декодер ---
        # Вход: латентный вектор (latent_dim)
        self.dec_fc1 = nn.Linear(latent_dim + text_embedding_dim, 256)
        self.dec_fc2 = nn.Linear(256, 256)
        self.dec_fc3 = nn.Linear(256, 3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, text_embedding):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = torch.cat([x, text_embedding], dim=1)
        mu = self.enc_fc_mu(x)
        logvar = self.enc_fc_logvar(x)
        z = self.reparameterize(mu, logvar)

         # --- Декодер ---
        z = torch.cat([z, text_embedding], dim=1)
        x = F.relu(self.dec_fc1(z))
        x = F.relu(self.dec_fc2(x))
        x = self.dec_fc3(x)
        return x, mu, logvar
    # --- Функция потерь для VAE ---
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss