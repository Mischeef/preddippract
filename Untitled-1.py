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