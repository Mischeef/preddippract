import os
import trimesh
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
import matplotlib.pyplot as plt

# Класс для загрузки данных ModelNet10
class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.data, self.categories = self._load_data()
        self.label_map = {label: idx for idx, label in enumerate(self.categories)}

    def _load_data(self):
        data = []
        categories = set()
        for category in os.listdir(self.root_dir):
            category_path = os.path.join(self.root_dir, category, self.mode)
            if os.path.exists(category_path):
                categories.add(category)
                for file in os.listdir(category_path):
                    if file.endswith('.off'):
                        data.append((os.path.join(category_path, file), category))
        return data, list(categories)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, category = self.data[idx]
        mesh = trimesh.load(file_path, force='mesh')
        vertices = mesh.vertices  # Получаем вершины модели
        label = self.label_map[category]
        if self.transform:
            vertices = self.transform(vertices)
        return vertices, label

# Функция для коллатного батча с паддингом
def collate_fn(batch):
    vertices, labels = zip(*batch)
    vertices_padded = pad_sequence([torch.tensor(v, dtype=torch.float32) for v in vertices], batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return vertices_padded, labels

# Укажите абсолютный путь к папке ModelNet10
modelnet10_path = 'c:/Users/gerpv/Desktop/predpp/preddippract/ModelNet10'

# Пример использования
train_dataset = ModelNet10Dataset(root_dir=modelnet10_path, mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

test_dataset = ModelNet10Dataset(root_dir=modelnet10_path, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Настройка модели и оптимизатора
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model('text300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Обучение модели и построение графиков
num_epochs = 5
losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for vertices, labels in train_dataloader:
        vertices = vertices.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = diffusion.training_losses(model, vertices, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_dataloader)
    losses.append(epoch_loss)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')

# Построение графика потерь
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
