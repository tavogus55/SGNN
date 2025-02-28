import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from torch.nn import Linear


# Define the SGC model
class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K=2)  # K=2 means using a 2-hop neighborhood

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.log_softmax(x, dim=1)


# Training function
def train(model, optimizer, device, train_loader):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])  # Now `batch.y` is 1D
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation function
def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            correct += (pred[batch.test_mask] == batch.y[batch.test_mask]).sum().item()
            total += batch.test_mask.sum().item()
    return correct / total
