import torch
import torch.nn as nn
import torch.optim as optim

def train(model, loader, device, epochs, lr):
    device = torch.device(device)
    model.to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0

        for x, y, _ in loader:
            x = x.to(device)
            y = y.float().to(device)        

            logits = model(x).view(-1)     
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
