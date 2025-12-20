import torch
import torch.nn as nn
import torch.optim as optim

def train(model, loader, device, epochs, lr):
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0

        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze()
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")
