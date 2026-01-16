import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from load_data import FRBDataset
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0001
REAL_DATA_PATH = "data/real_frbs"

train_ds = FRBDataset(mode='synthetic', num_synthetic=5000)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

val_ds = FRBDataset(mode='real', real_data_dir=REAL_DATA_PATH)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
if len(val_ds) == 0:
    print("no real files found.")


class DM_Predictor_CNN(nn.Module):
    def __init__(self):
        super(DM_Predictor_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(
                32), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(
                64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(
                128), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(
                256), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(
                512), nn.LeakyReLU(0.1), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.LeakyReLU(0.1), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


model = DM_Predictor_CNN().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()

loss_history = []
print("training now")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch+1) % 2 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")

torch.save(model.state_dict(), "best_frb_model.pth")
print("model saved")

if len(val_ds) > 0:
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for img, target in val_loader:
            img = img.to(DEVICE)
            p = model(img).item() * 1000.0
            a = target.item() * 1000.0
            preds.append(p)
            actuals.append(a)
            print(f"True: {a:.0f} vs Pred: {p:.0f}")

    mae = sum([abs(p - a) for p, a in zip(preds, actuals)]) / len(preds)

    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, preds, c='blue', alpha=0.6,
                edgecolors='k', label='Predictions')
    plt.plot([0, 2500], [0, 2500], 'r--', lw=2, label="Ideal Fit")
    plt.xlabel("True DM (pc cm^-3)")
    plt.ylabel("Predicted DM (pc cm^-3)")
    plt.title(f"Final Results - MAE: {mae:.2f}")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_results.png")
    plt.show()