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

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


def train(model, data, optimizer, epochs, train_mask=None, loader=None, logger=None):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        if loader is not None:
            # Minibatch training using NeighborLoader
            total_loss = 0
            for batch in loader:
                batch = batch.to(next(model.parameters()).device)  # Ensure correct device

                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)

                # Convert y to 1D tensor if necessary
                target = batch.y if batch.y.dim() == 1 else batch.y.argmax(dim=1)

                loss = F.cross_entropy(out[batch.train_mask], target[batch.train_mask])
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
        else:
            # Full-batch training
            out = model(data.x, data.edge_index)

            target = data.y if data.y.dim() == 1 else data.y.argmax(dim=1)
            loss = F.cross_entropy(out[train_mask], target[train_mask])

            loss.backward()
            optimizer.step()
            avg_loss = loss.item()

        if epoch % 10 == 0 and logger:
            logger.debug(f"Epoch {epoch}: Loss: {avg_loss:.4f}")


def test(model, data, test_mask, loader=None):
    model.eval()

    if loader is not None:
        # Minibatch inference
        correct = total = 0
        for batch in loader:
            batch = batch.to(next(model.parameters()).device)  # Ensure correct device
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            correct += pred[batch.test_mask].eq(batch.y[batch.test_mask]).sum().item()
            total += batch.test_mask.sum().item()
        acc = correct / total
    else:
        # Full-batch inference
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[test_mask].eq(data.y[test_mask]).sum().item()
        acc = correct / test_mask.sum().item()

    return acc
