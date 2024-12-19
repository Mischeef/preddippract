import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from shap_e.models.download import load_model, load_config
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
import matplotlib.pyplot as plt

# Выбрать директорию в которой находится датасет
os.chdir(r'C:\Users\gerpv\Desktop\predpp\preddippract')

# Определим класс для загрузки данных ModelNet10
class ModelNet10Dataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.classes = sorted(os.listdir(data_dir))
        self.data = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            subset_dir = os.path.join(class_dir, 'train' if self.train else 'test')
            if not os.path.isdir(subset_dir):
                continue
            for file_name in os.listdir(subset_dir):
                if file_name.endswith('.off'):
                    self.data.append(os.path.join(subset_dir, file_name))
                    self.labels.append(label)

        if len(self.data) == 0:
            raise ValueError(f"No data found in directory: {data_dir}")

        # Перемешивание данных
        combined = list(zip(self.data, self.labels))
        np.random.shuffle(combined)
        self.data, self.labels = zip(*combined)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        try:
            points = self.load_off_file(file_path)
        except ValueError as e:
            print(f"Skipping invalid file: {file_path}. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))  # Переход к следующему элементу

        if self.transform:
            points = self.transform(points)

        return points, label

    def load_off_file(self, file_path):
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            if 'OFF' != header:
                raise ValueError(f'File is not a valid OFF file: {file_path}')
            line = f.readline().strip().split()
            if len(line) != 3:
                raise ValueError(f'Invalid OFF file format: {file_path}')
            n_verts, n_faces, _ = map(int, line)
            verts = []
            for _ in range(n_verts):
                try:
                    verts.append(list(map(float, f.readline().strip().split())))
                except ValueError:
                    raise ValueError(f'Invalid vertex format in OFF file: {file_path}')
        return np.array(verts)

# Функция для коллатного батча с паддингом
def collate_fn(batch):
    vertices, labels = zip(*batch)
    max_vertices = max(v.shape[0] for v in vertices)
    vertices_padded = np.zeros((len(vertices), max_vertices, 3), dtype=np.float32)
    for i, v in enumerate(vertices):
        vertices_padded[i, :v.shape[0], :] = v
    vertices_padded = torch.tensor(vertices_padded)
    labels = torch.tensor(labels, dtype=torch.long)
    return vertices_padded, labels

# Укажите абсолютный путь к папке ModelNet10
data_dir = r'C:\Users\gerpv\Desktop\predpp\preddippract\ModelNet10'

# Пример использования
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ModelNet10Dataset(data_dir=data_dir, transform=transform, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

test_dataset = ModelNet10Dataset(data_dir=data_dir, transform=transform, train=False)
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
