import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F


class SGC(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(SGC, self).__init__()
        self.conv = SGConv(num_features, num_classes, K=2, cached=False)  # Single-layer SGC

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


def train(model, loader, optimizer, epochs, device, logger=None):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if logger:
            logger.debug(f"Epoch {epoch + 1}: Loss: {total_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}: Loss: {total_loss:.4f}")


@torch.no_grad()
def test(model, data, device):
    model.eval()
    out = model(data.x, data.edge_index).argmax(dim=1)
    correct = (out[data.test_mask] == data.y[data.test_mask]).sum().item()
    total = data.test_mask.sum().item()
    return correct / total
