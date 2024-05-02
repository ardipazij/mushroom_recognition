import torchvision
import torch
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from zipfile import ZipFile
import random
import PIL
from PIL import Image
import os
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWrite

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

input_dir = 'content/Mushroom-recognition'
batch_size = 32
rescale_size = 64
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((int(rescale_size), int(rescale_size))),
    transforms.RandomHorizontalFlip(),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.8),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_full = torchvision.datasets.ImageFolder(root=input_dir, transform=transform)

# split full dataset
train_idx, valid_idx = train_test_split(list(range(len(dataset_full))), train_size=0.9)
dataset = {
    'train': torch.utils.data.Subset(dataset_full, train_idx),
    'valid': torch.utils.data.Subset(dataset_full, valid_idx)
}

dataset_size = {ds: len(dataset[ds]) for ds in ['train', 'valid']}
dataset_classes = np.array(dataset_full.classes)

dataloader = {
    'train': torch.utils.data.DataLoader(
        dataset=dataset['train'], batch_size=batch_size, shuffle=True
    ),
    'valid': torch.utils.data.DataLoader(
        dataset=dataset['valid'], batch_size=batch_size, shuffle=False
    ),
}

id2class = {i: cls for i, cls in enumerate(dataset_classes)}

tensorboard_writer = SummaryWriter('./tensorboard_logs')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

from collections import namedtuple
from typing import NamedTuple, List

EvalOut = namedtuple("EvalOut", ['loss', 'accuracy'])
os.makedirs('content/mushroom_logs', exist_ok=True)

def eval_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss._Loss,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    acc_loss = 0
    accuracy = 0
    total = len(loader.dataset)
    model.eval()
    model.to(device)
    with torch.inference_mode():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            acc_loss += loss.item()
            accuracy += torch.sum(torch.argmax(pred, 1) == target).item()

    return EvalOut(loss = (acc_loss / total), accuracy = (accuracy / total))


class TrainOut(NamedTuple):
    train_loss: List[float]
    eval_loss: List[float]
    eval_accuracy: List[float]


def train(
    model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss._Loss,
    sheduler: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 10
):
    train_loss = []
    eval_loss = []
    eval_accuracy = []
    model.to(device)
    for i in range(epochs):
        print(f"Epoch - {i}\n")
        if (train_loader != None):
            print("Train...\n")
            train_loss.append(train_epoch(model, optimizer, criterion, train_loader, device))
            tensorboard_writer.add_scalar('loss/training', train_loss[-1], i)
        print("Validation...\n")
        eval_out = eval_epoch(model, criterion, val_loader, device)
        eval_loss.append(eval_out.loss)
        eval_accuracy.append(eval_out.accuracy)
        tensorboard_writer.add_scalar('validation accuracy', eval_out.accuracy, i)
        tensorboard_writer.add_scalar('loss/validation', eval_out.loss, i)
        print(f'Validation acc: {eval_out.accuracy}')
        sheduler.step()
        print('lr: ', get_lr(optimizer))
        if i > 1 and eval_accuracy[i] == max(eval_accuracy) and i % 3 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }, f'content/mushroom_logs/epoch_{i}.pth')

    return TrainOut(train_loss = train_loss,
                    eval_loss = eval_loss,
                    eval_accuracy = eval_accuracy), model



def show_losses(TrainOut, epochs):
    plt.plot(epochs, TrainOut.train_loss)
    plt.plot(epochs, TrainOut.eval_loss)
    plt.show()

def show_accuracy(accuracy, epochs):
    plt.plot(epochs, accuracy)
    plt.show()

def predict(model, dataloader_test):
    logits = []
    model.eval()
    with torch.no_grad():
        for inputs, _ in dataloader_test:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu()
            logits.append(outputs)
    probs = torch.nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs

import torchvision.transforms as T
from torchvision.models import mobilenet_v3_large
mobilenet = mobilenet_v3_large(weights='IMAGENET1K_V2')

class MobileNet(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.mobilenet = mobilenet_v3_large(weights='IMAGENET1K_V2')
    for param in self.mobilenet.parameters():
      param.requires_grad = False

    for j in range(15, 17):
      for param in self.mobilenet.features[j].parameters():
        param.requires_grad = True

    for param in self.mobilenet.classifier.parameters():
      param.requires_grad = True

    self.mobilenet.classifier[3] = torch.nn.Linear(1280, 20)

    self.transforms =  torch.nn.Sequential(
            T.Resize(224),  # We use single int value inside a list due to torchscript type restrictions
#            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x = self.transforms(x)
    y = self.mobilenet(x)
    return y

from tqdm.notebook import tqdm

def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    criterion: torch.nn.modules.loss._Loss,
    loader: torch.utils.data.DataLoader,
    device: torch.device
):
    acc_loss = 0
    total = len(loader.dataset)
    model.to(device)
    model.train()
    for data, target in tqdm(loader):
      # with accelerator.accumulate(model): # для имитации большого размера батча (полезно для трансформеров)
        data = data.to(device)
        target = target.to(device)
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc_loss += loss.item()
    for n, (img, pred, label) in enumerate(zip(data, pred, target)):
        if n == 31:
            tensorboard_writer.add_image("testing/{}_GT_{}_pred_{}"
                                          .format(n, id2class[int(label)], id2class[int(torch.argmax(pred, 1))]), img)

    return acc_loss / total

model = MobileNet().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
sheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
epochs = 40
configuration_dict = {'number_of_epochs': epochs, 'batch_size': batch_size, 'base_lr': 1e-4, 'weight_decay': 1e-4, 'rescale_size': 64}
tr_tuple, model = train(model, optimizer, criterion, sheduler, dataloader['train'], dataloader['valid'], device, epochs)

show_losses(tr_tuple, range(100))

!pip install rembg
from rembg import remove

class ModelInference:
    def __init__(self, model_path: str, device='cpu'):
        self.device = device
        self.model = MobileNet()
        checkpoint = torch.load(model_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def _remove_background(self, img_path):
        pil_img = Image.open(img_path).convert('RGB')

        # Удаление фона с помощью rembg
        img_without_bg = remove(pil_img).convert('RGB')
        display(img_without_bg)
        return img_without_bg

    def inference(self, img_path='sample.jpg'):
        pil_img = self._remove_background(img_path)
        tensor = self.transforms(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)
            _, predicted = torch.max(preds, 1)
        return id2class[predicted.item()]
