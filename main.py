import torch
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from dataset.kadid import KadidDataset
from torch.utils.data import DataLoader
from model.trcnn import TrCNN
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import os
load_dotenv()

CSV_PATH = os.getenv("KADID_CSV_PATH")
IMAGE_PATH = os.getenv("KADID_IMAGE_PATH")

data = KadidDataset(
  csv_file=CSV_PATH,
  root_dir=IMAGE_PATH,
  transform=ToTensor()
)

loader = DataLoader(data, batch_size=100, shuffle=True, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

trcnn = TrCNN(32, 32, 10, 1).to(device)

EPOCHS = 100

optimizer = Adam(trcnn.parameters(), lr=0.005)
criterion = nn.MSELoss()

if __name__ == "__main__":

  for epoch in range(EPOCHS):
    for batch in tqdm(loader, desc="Training"):
      images, labels = batch
      images, labels = images.to(device).type(torch.float32), labels.to(device).type(torch.float32)

      pred = trcnn(images)

      loss = criterion(pred.squeeze(1), labels)
      print(loss)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(f'Epoch {epoch + 1}/{EPOCHS} loss: {loss.item() :.3f}')