import torch

import torch.optim as optim


def calculate_corrects(last_outputs, last_labels):
  predicted = torch.where(last_outputs > 0.5, 1, torch.where(last_outputs < -0.5, -1, 0))
  corrects = (predicted == last_labels).sum().item()
  return corrects

def validate_model(model, valid_dl, loss_func1, loss_func2, device):
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):

            images, labels = images.to(device), labels.to(device)
            # labels = labels.squeeze()

            # Forward pass ➡
            outputs, rnn_input, pred_out  = model(images)
            last_outputs = outputs[:,-5:]

            val_loss += loss_func1(last_outputs, labels)*labels.size(0)
            val_loss += loss_func2(rnn_input, pred_out)
            # Compute accuracy and accumulate
            correct += calculate_corrects(outputs[:,-1], labels[:,-1])

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

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
