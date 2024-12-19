import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split

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

# Определим простую нейронную сеть
class SimplePointNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimplePointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Загрузка данных
data_dir = 'ModelNet10'
try:
    train_dataset = ModelNet10Dataset(data_dir, train=True)
    test_dataset = ModelNet10Dataset(data_dir, train=False)
except ValueError as e:
    print(e)
    exit(1)

# Функция для обработки батчей
def collate_fn(batch):
    points, labels = zip(*batch)
    points = [torch.tensor(p, dtype=torch.float32) for p in points]
    labels = torch.tensor(labels, dtype=torch.long)
    return points, labels

# Выбор batch_size
batch_size = 32
num_epochs = 20

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Инициализация модели, функции потерь и оптимизатора
model = SimplePointNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Обучение модели
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Перенос данных на устройство
        inputs = [x.permute(1, 0).unsqueeze(0).to(device) for x in inputs]
        labels = labels.long().to(device)

        optimizer.zero_grad()
        outputs = torch.cat([model(x) for x in inputs])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Печать каждые 10 батчей
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}')
            running_loss = 0.0

    # Валидация модели
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = [x.permute(1, 0).unsqueeze(0).to(device) for x in inputs]
            labels = labels.long().to(device)
            outputs = torch.cat([model(x) for x in inputs])
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    # Сохранение лучшей модели
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        model_path = os.path.join(os.getcwd(), 'best_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

print('Finished Training')