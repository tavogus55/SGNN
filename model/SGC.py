import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F

class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, 128)
        self.conv3 = SGConv(128, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


def train(model, loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)  # Move batch to device
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}: Loss: {total_loss:.4f}")


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        batch = batch.to(device)  # Move batch to device
        out = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        correct += pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item()
        total += batch.test_mask.sum().item()
    return correct / total
