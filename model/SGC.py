import torch
from torch_geometric.nn import SGConv
import torch.nn.functional as F
from torch.nn import Linear


class SGC(torch.nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        """
        Two-layer SGC model:
        - First layer: size = hidden_size (128 as per paper).
        - Second layer: size = num_classes (for classification tasks).
        """
        super(SGC, self).__init__()
        self.conv1 = SGConv(num_features, hidden_size)
        self.conv2 = SGConv(hidden_size, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


def train(model, data, optimizer, epochs, train_mask, logger=None):
    model.train()
    for epoch in range(epochs):  # Number of epochs
        optimizer.zero_grad()
        out = model(data)
        # Compute loss using only training nodes
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch}: Loss: {loss.item():.4f}")


def test(model, data, test_mask):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)  # Predicted clasgcs for each node
    correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
    acc = correct / test_mask.sum().item()
    return acc
