import torch

import torch.optim as optim


def calculate_corrects(last_outputs, last_labels):
  predicted = torch.where(last_outputs > 0.5, 1, torch.where(last_outputs < -0.5, -1, 0))
  corrects = (predicted == last_labels).sum().item()
  return corrects

def validate_model(model, valid_dl, loss_func1, loss_func2, device):
    model.eval()
    val_loss1 = 0.
    val_loss2 = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):

            images, labels = images.to(device), labels.to(device)
            # labels = labels.squeeze()

            # Forward pass ➡
            outputs, rnn_input, pred_out  = model(images)
            last_outputs = outputs[:,-5:]

            val_loss1 += loss_func1(last_outputs, labels)*labels.size(0)
            val_loss2 += loss_func2(rnn_input, pred_out)*labels.size(0)
            # Compute accuracy and accumulate
            correct += calculate_corrects(outputs[:,-1], labels[:,-1])

    return val_loss1 / len(valid_dl.dataset), val_loss2 / len(valid_dl.dataset), correct / len(valid_dl.dataset)

def copy_weights(model, alexnet):
    model.conv1.weight.data = alexnet.features[0].weight.data
    model.conv1.bias.data = alexnet.features[0].bias.data
    model.conv2.weight.data = alexnet.features[3].weight.data
    model.conv2.bias.data = alexnet.features[3].bias.data
    model.conv3.weight.data = alexnet.features[6].weight.data
    model.conv3.bias.data = alexnet.features[6].bias.data
    model.conv4.weight.data = alexnet.features[8].weight.data
    model.conv4.bias.data = alexnet.features[8].bias.data
    model.conv5.weight.data = alexnet.features[10].weight.data
    model.conv5.bias.data = alexnet.features[10].bias.data

def compute_var(embeddings):
    embeddings_flat = embeddings.reshape(-1, embeddings.size(2))  # New size: (N*L, D)

    # Compute mean across the batch (considering each time step in the sequence as part of the batch)
    mean_per_dim = embeddings_flat.mean(dim=0)  # Size: (D,)

    # Compute variance across the batch for each dimension
    variance_per_dim = ((embeddings_flat - mean_per_dim) ** 2).mean(dim=0)  # Size: (D,)

    # Apply the threshold γ (gamma) to the standard deviation, not variance
    gamma = 1.0  # Example threshold for standard deviation
    std_per_dim = torch.sqrt(variance_per_dim)  # Standard deviation for each dimension
    variance_loss = torch.relu(gamma - std_per_dim).mean()  # Applying the VICReg variance component formula
    return variance_loss



def validate_model_reg(model, valid_dl, loss_func, device):
    model.eval()
    val_loss1 = 0.
    val_loss2 = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):

            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze()

            outputs = model(images)

            val_loss1 += loss_func(outputs, labels)*labels.size(0)
            # Compute accuracy and accumulate
            correct += calculate_corrects(outputs[:,-1], labels[:,-1])

    return val_loss1 / len(valid_dl.dataset), correct / len(valid_dl.dataset)

