#Brevitas testing script for Quantized LSTM

import torch
import math
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import brevitas.nn as bnn
from brevitas.nn import QuantLinear, QuantReLU
from brevitas.quant import SignedBinaryWeightPerTensorConst, Uint8ActPerTensorFloat, Int8ActPerTensorFloat
import torch.nn as nn
import itertools
import time
import os
from scipy.spatial.distance import hamming

# ------Setting the training device-------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ------ DatasetLoader -------------------------

class QLSTMTestDataset(Dataset):
    def __init__(self):
        # will be mostly used for dataloading
        x_load = np.loadtxt('../train_preparation/4_attack_lstm_test_x.txt',delimiter=",", dtype=np.float32)
        y_load = np.loadtxt('../train_preparation/4_attack_lstm_test_y_multiclass.txt', delimiter=",", dtype=np.float32)
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class QLSTMIDS(nn.Module):
    def __init__(self):
        super(QLSTMIDS, self).__init__()
        # To be uncommented while training an LSTM model.
        self.qlstm = bnn.QuantLSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True,
                                   weight_bit_width=8,
                                   io_quant=Int8ActPerTensorFloat,
                                   gate_acc_bit_width=6,
                                   sigmoid_bit_width=6,
                                   tanh_bit_width=6,
                                   cell_state_bit_width=6,
                                   bias_quant=None)
        self.qfc1 = bnn.QuantLinear(20, 64, bias=True, weight_bit_width=8)
        self.qfc2 = bnn.QuantLinear(64, 32, bias=True, weight_bit_width=8)
        self.qfc3 = bnn.QuantLinear(32, 5, bias=True, weight_bit_width=8)
        self.qrelu = bnn.QuantReLU(bit_width=6)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, batch_size):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, batch_size, 20).requires_grad_().to("cpu")
        # Initialize cell state
        c0 = torch.zeros(1, batch_size, 20).requires_grad_().to("cpu")
        # Start model definition
        out, (hn, cn) = self.qlstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(hn[-1, :, :])
        out = self.qrelu(out)
        out = self.qfc1(out)
        out = self.qrelu(out)
        out = self.qfc2(out)
        out = self.qrelu(out)
        out = self.qfc3(out)
        return out


testDataset = QLSTMTestDataset()
first_data = testDataset[0]
features, labels = first_data
print(features, labels)
samples = len(testDataset)

test_batch_size = 1400
testloader = DataLoader(dataset=testDataset,batch_size=test_batch_size, shuffle=False)

model = QLSTMIDS()
model = model.float()
max_accuracy = 0
max_index = 100
acc = 0
count_norm = 0
count_norm_acc = 0
count_atk = 0
count_atk_acc = 0
full_test_label = []
full_pred_label = []
for j in range(100):
    path = './lstm_multiclass_pruned_v1/model_0/models/model_'+str(j)+'.pt' 
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    count = np.zeros(7)
    model.eval()
    t1 = 0
    t2 = 0
    t3 = 0
    acc = 0
    seq_len = 2
    input_length = 10
    count_norm = 0
    count_norm_acc = 0
    count_atk = 0
    count_atk_acc = 0
    count_atk_f = 0
    count_atk_f_acc = 0
    count_atk_r = 0
    count_atk_r_acc = 0
    count_atk_g = 0
    count_atk_g_acc = 0
    print_count = 0
    accuracy = 0  # We report the accuracy of the model here.
    for l, (test_inputs, test_labels) in enumerate(testloader):
        t1 = t1 + time.time()
        test_inputs = test_inputs.reshape(
            [test_batch_size, seq_len, input_length])
        outputs = model(test_inputs.float(), test_batch_size)
        a = outputs.detach().numpy()
        a = a.reshape([test_batch_size, 5])
        a = np.round(a)
        b = test_labels.detach().numpy()
        # print(b)
        max_idx_pred = a.argmax(axis=1)  # .argmax(axis=1)
        max_idx_test = b.argmax(axis=1)
        for i in range(test_batch_size):
            if (max_idx_test[i] == max_idx_pred[i]):
                acc = acc+1
            if (max_idx_test[i] == 0):
                count_norm = count_norm+1
                if (max_idx_pred[i] == 0):
                    count_norm_acc = count_norm_acc + 1
                else:
                    print(max_idx_pred[i])
            if (max_idx_test[i] == 1):
                count_atk = count_atk+1
                if (max_idx_pred[i] == 1):
                    count_atk_acc = count_atk_acc + 1
            if (max_idx_test[i] == 2):
                count_atk_f = count_atk_f+1
                if (max_idx_pred[i] == 2):
                    count_atk_f_acc = count_atk_f_acc + 1
                else:
                    print("Fuzzy : ",max_idx_pred[i])
            if (max_idx_test[i] == 3):
                count_atk_r = count_atk_r+1
                if (max_idx_pred[i] == 3):
                    count_atk_r_acc = count_atk_r_acc + 1
            if (max_idx_test[i] == 4):
                count_atk_g = count_atk_g+1
                if (max_idx_pred[i] == 4):
                    count_atk_g_acc = count_atk_g_acc + 1
        accuracy = accuracy + (test_batch_size-hamming(max_idx_pred, max_idx_test)*len(max_idx_test))
        t2 = t2 + time.time()
    if (accuracy > max_accuracy):
        max_accuracy = accuracy
        max_index = j
    t3 = t3+t2-t1
    print('Total messages =', samples, 'Overall accuracy =', accuracy, 'Misclassifications = ', (samples-accuracy), 'Percentage accuracy =',
          (accuracy/samples)*100, 'Epoch =', int(j), '\n')  # Print the accuracy and the percentage accuracy here in this statement.
    print('Total Normal =', count_norm, ' Correct normal =', count_norm_acc,
          'Misclassifications = ', (count_norm-count_norm_acc), '\n')
    print('Total Attack DoS =', count_atk, ' Correct Attack DoS =',
          count_atk_acc, 'Misclassifications = ', (count_atk-count_atk_acc), '\n')
    print('Total Attack Fuzzy =', count_atk_f, ' Correct Attack Fuzzy =',
          count_atk_f_acc, 'Misclassifications = ', (count_atk_f-count_atk_f_acc), '\n')
    print('Total Attack RPM =', count_atk_r, ' Correct Attack RPM =',
          count_atk_r_acc, 'Misclassifications = ', (count_atk_r-count_atk_r_acc), '\n')
    print('Total Attack Gear =', count_atk_g, ' Correct Attack Gear =',
          count_atk_g_acc, 'Misclassifications = ', (count_atk_g-count_atk_g_acc), '\n')
    print('---------------------------')
print("Maximum accuracy index = ", max_index,"Max accuracy = ", (max_accuracy/samples)*100)