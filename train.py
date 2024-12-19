import os
import trimesh
from torch.utils.data import Dataset, DataLoader
import torch
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
import matplotlib.pyplot as plt

# Класс для загрузки данных ModelNet10
class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.data = self._load_data()

    def _load_data(self):
        data = []
        categories = os.listdir(self.root_dir)
        for category in categories:
            category_path = os.path.join(self.root_dir, category, self.mode)
            if os.path.exists(category_path):
                for file in os.listdir(category_path):
                    if file.endswith('.off'):
                        data.append((os.path.join(category_path, file), category))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, category = self.data[idx]
        mesh = trimesh.load(file_path, force='mesh')
        vertices = mesh.vertices  # Получаем вершины модели
        if self.transform:
            vertices = self.transform(vertices)
        return vertices, category

# Укажите абсолютный путь к папке ModelNet10
modelnet10_path = 'c:/Users/gerpv/Desktop/predpp/preddippract/ModelNet10'

# Пример использования
train_dataset = ModelNet10Dataset(root_dir=modelnet10_path, mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = ModelNet10Dataset(root_dir=modelnet10_path, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

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
    for vertices, categories in train_dataloader:
        # Преобразование данных в формат, подходящий для модели
        vertices = torch.tensor(vertices, dtype=torch.float32).to(device)
        categories = torch.tensor(categories).to(device)
        optimizer.zero_grad()
        loss = diffusion.training_losses(model, vertices, categories)
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
