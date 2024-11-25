import torch
import os
from torch.optim import Adam
import torch.nn as nn
import pickle
from tqdm import tqdm
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from dataset.kadid import KadidDataset
from torch.utils.data import DataLoader
from model.trcnn import TrCNN
import os
load_dotenv()

MODEL_PATH = "models"
LOSS_HISTORY_FOLDER = "loss_history"
EPOOCHS = 10
SAVE_EACH_BATCH = False
SAVE_EACH_EPOCH = True
CSV_PATH = os.getenv("KADID_CSV_PATH")
IMAGE_PATH = os.getenv("KADID_IMAGE_PATH")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train(trcnn, loader, epoch, learning_rate=0.01, device='cpu'):

  optimizer = Adam(trcnn.parameters(), lr=0.005)
  criterion = nn.MSELoss()

  loss_history_path = LOSS_HISTORY_FOLDER + "/epoch-" + str(epoch) + ".pkl"

  if not os.path.exists(LOSS_HISTORY_FOLDER):
    os.makedirs(LOSS_HISTORY_FOLDER)

  if os.path.exists(loss_history_path):
    with open(loss_history_path, 'rb') as f:
      loss_history = pickle.load(f)
  
  total_loss = 0
  loss_history = []
  
  for batch in tqdm(loader, desc="Training"):
    images, labels = batch
    images, labels = images.to(device).type(torch.float32), labels.to(device).type(torch.float32)

    pred = trcnn(images)

    loss = criterion(pred.squeeze(1), labels)
    print("Batch Loss :" + loss)

    loss_history.append(loss.item())
    with open(loss_history_path, 'wb') as f:
      pickle.dump(loss_history, f)

    total_loss += loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  mean_loss = total_loss / len(loader)
  return mean_loss

data = KadidDataset(
  csv_file=CSV_PATH,
  root_dir=IMAGE_PATH,
  transform=ToTensor()
)

loader = DataLoader(data, batch_size=100, shuffle=True, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device: ", device)

trcnn = TrCNN(32, 32, 10, 1).to(device)

trcnn.load_model(MODEL_PATH)

if __name__ == '__main__':
  torch.cuda.empty_cache()
  for epoch in range(EPOOCHS):
    loss = train(trcnn, loader, device=device, epoch=epoch)
    print(f"Epoch: {epoch}, Loss: {loss}")
    if SAVE_EACH_BATCH:
      trcnn.save_model(MODEL_PATH)
  if SAVE_EACH_EPOCH:
    trcnn.save_model(MODEL_PATH)

  
