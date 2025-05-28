import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import ViTWaypointPredictor
from Dataset import WaymoTrajectoryDataset, train_dataset_iter, val_dataset_iter
loss_list = []

def train(model, dataset, device='cpu', epochs=2, batch_size=16):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Only trainable parameters (ViT is frozen)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch["images"].to(device)        # [B, 3, 3, 224, 224]
            intent = batch["intent"].to(device)        # [B, 4]
            past_dyn = batch["past_dyn"].to(device)    # [B, 16, 6]
            targets = batch["target"].to(device)       # [B, 20, 3]

            optimizer.zero_grad()
            preds = model(images, intent, past_dyn)    # [B, 20, 2]

            loss = loss_fn(preds, targets[:, :, :2])   # compare only (x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        loss_list.append(total_loss)

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(dataloader):.4f}")



train_dataset = WaymoTrajectoryDataset(train_dataset_iter)
val_dataset = WaymoTrajectoryDataset(val_dataset_iter, n_samples=10)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
# Test a batch
# for batch in train_loader:
#     print(batch["images"].shape)    # [8, 3, 3, 224, 224]
#     print(batch["intent"].shape)    # [8, 4]
#     print(batch["past_dyn"].shape)  # [8, 16, 6]
#     print(batch["target"].shape)    # [8, 20, 3]
#     break
model = ViTWaypointPredictor()
# print the total params in model
print("Total parameters in model:", sum(p.numel() for p in model.parameters() if p.requires_grad))

train(model, dataset=train_dataset, epochs = 10)
# plot the loss vs epochs
# fig, ax = plt.subplots()
# ax.plot(loss_list)
# ax.set_xlabel('Epochs')
# ax.set_ylabel('Loss')
# ax.set_title('Loss vs Epochs')
# plt.show()