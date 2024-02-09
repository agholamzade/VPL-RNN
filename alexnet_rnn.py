import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNetRNN(nn.Module):
    def __init__(self):
        super(AlexNetRNN, self).__init__()
        self.rnn_input = 32
        self.hidden_size = 64

        self.seq_len = 10
        self.added_zeros = 5

        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)

        self.fc1 = nn.Linear(580416, self.rnn_input) # for copied weights
        #self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.MaxPool2d(3, stride=2)

        self.rnn = nn.GRU(self.rnn_input, self.hidden_size, 1, batch_first=True)

        self.fc2 = nn.Linear(self.hidden_size, 1)

        self.fc3 = nn.Linear(self.hidden_size, self.rnn_input)


    def forward(self, x1):

      x1 = x1.view(-1, 3, 227,227)
      skip_x1 = []
      # x1
      x1 = F.relu(self.conv1(x1))
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,64,56,56]

      x1 = self.pool(x1)
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,64,27,27]

      x1 = F.relu(self.conv2(x1))
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,192,27,27]

      x1 = self.pool(x1)
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,192,13,13]

      x1 = F.relu(self.conv3(x1))
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,384,13,13]

      x1 = F.relu(self.conv4(x1))
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,256,13,13]

      x1 = F.relu(self.conv5(x1))
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # [1,256,13,13]

      x1 = self.pool(x1)
      x1_dim = x1.shape
      skip_x1.append(x1.view(-1, x1_dim[1] * x1_dim[2] * x1_dim[3])) # 1,256,6,6]

      x1 = torch.cat(skip_x1, dim=1)

      # save activation
      #torch.save(x1, save_dir+'x1_target_final_activation_sep_'+str(sep)+".pt")

      x1 = self.fc1(x1)
      x1 = F.relu(x1)

      x1 = x1.view(-1, self.seq_len, self.rnn_input)

      batch_size = x1.shape[0]

      rnn_input = x1.clone()
      rnn_input = x1[:,1:,:]

      zeros_to_concat = torch.zeros(batch_size, self.added_zeros, self.rnn_input, device=x1.device)
      x1 = torch.cat((x1, zeros_to_concat), dim=1)

      h0 = torch.zeros(1, batch_size, self.hidden_size, device=x1.device)
      # c0 = torch.zeros_like(h0)

      out, h = self.rnn(x1, h0)

      pred_out = self.fc3(out)
      pred_out =  F.relu(pred_out)
      pred_out = pred_out[:,0:9,:]

      out = self.fc2(out)
      out = F.sigmoid(out)

      return out.squeeze(2), rnn_input, pred_out