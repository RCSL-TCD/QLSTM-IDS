#Breviatas training script for QLSTM model.

import torch
import math
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import brevitas.nn as bnn
from brevitas.quant import Int8ActPerTensorFloat,Uint8ActPerTensorFloat
import itertools
import time
import os
from torchsummary import summary

# ------Setting the training device-------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ------ Dataset loader---------------------------------------

class QLSTMTrainDataset(Dataset):
    def __init__(self):
        x_load = np.loadtxt('../train_preparation/4_attack_lstm_train_x.txt', delimiter=",", dtype=np.float32)
        y_load = np.loadtxt('../train_preparation/4_attack_lstm_train_y_multiclass.txt', delimiter=",", dtype=np.float32)
        #x_load = x_load - 128
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # Will allow us to get the length of our dataset
        return self.n_samples


class QLSTMValDataset(Dataset):
    def __init__(self):
        x_load = np.loadtxt('../train_preparation/4_attack_lstm_val_x.txt', delimiter=",", dtype=np.float32)
        y_load = np.loadtxt('../train_preparation/4_attack_lstm_val_y_multiclass.txt', delimiter=",", dtype=np.float32)
        #x_load = x_load - 128
        self.x = torch.from_numpy(x_load)
        self.y = torch.from_numpy(y_load)
        self.n_samples = x_load.shape[0]
        print("X_Load shape is = ",len(x_load))

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        # Will allow us to get the length of our dataset
        return self.n_samples


trainDataset = QLSTMTrainDataset()
valDataset = QLSTMValDataset()
first_data = trainDataset[0]
features, labels = first_data
print(features, labels)
print("Here = ",features.size())

train_batch_size = 2000 
val_batch_size = 2800 
# We train the model by setting shuffle as true to make batch training stable.
trainloader = DataLoader(dataset=trainDataset,batch_size=train_batch_size, shuffle=True)
validationloader = DataLoader(dataset=valDataset, batch_size=val_batch_size, shuffle=False)
num_epochs = 50
total_samples = len(trainDataset)
batches = math.ceil(total_samples/train_batch_size)
print(total_samples, batches)

running_loss = 0.0
running_val_loss = 0.0
valid_loss = np.zeros(num_epochs)
avg_valid_loss = np.zeros(num_epochs)
epoch_loss = np.zeros(num_epochs)
avg_epoch_loss_per_batch = np.zeros(num_epochs)
epoch = 0
count_val = 0
min_val_index = np.zeros(5)
lstm_weight_bit_width = 8
linear_weight_bit_width = 8
lstm_activation_bit_width = 6
linear_activation_bit_width = 6
num_outputs = 5

start = time.time()
for i in range(1):
    # ----- Model definition ------------------------
    base_directory = "./lstm_multiclass_pruned_v1"
    directory = "./lstm_multiclass_pruned_v1/model_"+str(i)
    data_folder = "./lstm_multiclass_pruned_v1/model_"+str(i)+"/data"
    model_folder = "./lstm_multiclass_pruned_v1/model_"+str(i)+"/models"
    os.mkdir(base_directory)
    os.mkdir(directory)
    os.mkdir(data_folder)
    os.mkdir(model_folder)

    class QLSTMIDS(nn.Module):
        def __init__(self):
            super(QLSTMIDS, self).__init__()
            # To be uncommented while training an LSTM model.
            self.qlstm = bnn.QuantLSTM(input_size=10, hidden_size=20,num_layers=1,batch_first=True,
                weight_bit_width=lstm_weight_bit_width,
                io_quant=Int8ActPerTensorFloat,
                gate_acc_bit_width=lstm_activation_bit_width,
                sigmoid_bit_width=lstm_activation_bit_width,
                tanh_bit_width=lstm_activation_bit_width,
                cell_state_bit_width=lstm_activation_bit_width,
                bias_quant=None)#Setting batch_first to "True" changed everything, Need to investigate why it worked.
            self.qfc1 = bnn.QuantLinear(20, 64,bias=True, weight_bit_width=linear_weight_bit_width)
            self.qfc2 = bnn.QuantLinear(64, 32,bias=True, weight_bit_width=linear_weight_bit_width)
            self.qfc3 = bnn.QuantLinear(32, 5,bias=True, weight_bit_width=linear_weight_bit_width)
            self.relu = nn.ReLU()
            self.qrelu = bnn.QuantReLU(bit_width=linear_activation_bit_width)
            self.dropout = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(20)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(32)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x,batch_size):
            # Initialize hidden state with zeros
            h0 = torch.zeros(1,batch_size, 20).requires_grad_().to("cuda:0")
            # Initialize cell state
            c0 = torch.zeros(1,batch_size, 20).requires_grad_().to("cuda:0")    
            #Start model definition
            out,(hn,cn) = self.qlstm(x,(h0.detach(),c0.detach())) 
            out = hn[-1, :, :]
            out = self.qrelu(out)
            out = self.qfc1(out)
            out = self.qrelu(out)
            out = self.qfc2(out)
            out = self.qrelu(out)
            out = self.qfc3(out)
            return out

    model = QLSTMIDS().to(device)
    #No need to load a floating point pre-trained model. QAT is good enough.
    seq_len = 2
    input_length = 10
    model = model.float()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    # Number of parameters in the model.
    print("No. of parameters in the model = ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Loop for training and validation of for a given iteration for a given number of epochs.
    h_0, c_0 = None, None
    for j in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_val_loss = 0.0
        count = 0
        epoch = 0
        count_test = 0
        epoch_time_start = time.time()
        for k, (inputs, labels) in enumerate(trainloader):
            start = time.time()
            inputs = inputs.reshape([train_batch_size,seq_len,input_length])
            inputs = inputs.to(device)
            outputs = model(inputs,train_batch_size)#The whole batch of 1024 inputs goes here together.
            outputs = outputs.cpu()
            labels = labels.reshape([train_batch_size, num_outputs])
            outputs = outputs.reshape([train_batch_size, num_outputs])
            #print(outputs)
            loss = criterion(outputs, labels)
            # Compute the gradients...derviate of the loss w.r.t the inputs
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            # Adjust the gradients.
            optimizer.step()
            count = count+1
            running_loss += loss.item()
            end = time.time()
            #print("Epoch and batch number training : ",j,k,(end-start))
        # Save the model for this particular epoch after training.
        model.eval()
        # The validation loop for the models to be trained here.
        for l, (val_inputs, val_labels) in enumerate(validationloader):
            #val_inputs = val_inputs.reshape([seq_len,val_batch_size,input_length])
            val_inputs = val_inputs.reshape([val_batch_size,seq_len,input_length])
            val_inputs = val_inputs.to(device)
            val_outputs = model(val_inputs,val_batch_size)
            val_outputs = val_outputs.cpu()
            val_labels = val_labels.reshape([val_batch_size, num_outputs])
            val_outputs = val_outputs.reshape([val_batch_size, num_outputs])
            val_loss = criterion(val_outputs, val_labels)
            count_val = count_val + 1
            running_val_loss += val_loss.item()
        epoch = epoch+1
        epoch_loss[j] = running_loss
        avg_epoch_loss_per_batch[j] = running_loss/batches
        valid_loss[j] = running_val_loss
        avg_valid_loss[j] = running_val_loss/val_batch_size
        print('Epoch Number = ', j, 'Epoch loss = ', running_loss, 'Average Epoch loss = ', running_loss/batches)
        print('Epoch Number = ', j, 'Epoch_Val loss = ', running_val_loss, 'Average Epoch_Val loss = ', running_val_loss/val_batch_size)
        epoch_time_end = time.time()
        total_epoch_time = epoch_time_end - epoch_time_start
        print("Total epoch time = "+str(total_epoch_time))
        print('--------------------------')
        path = model_folder+'/model_'+str(j)+'.pt'
        # Saving the model after every epoch
        torch.save(model.state_dict(), path)
    # Printing the best model of each configuration based on the Validation loss
    print('----------------------------------------------------Best model index based on validation loss = ', np.argmin(valid_loss))
    min_val_index[i] = np.argmin(valid_loss)
    # Saving the entire (normal and average) epoch and validation loss for all the epoch of a particular model here.
    path1 = data_folder+"/avg_el_"+str(i)+".txt"
    file = open(path1, "w+")
    content = str(avg_epoch_loss_per_batch)
    file.write(content)
    file.close()
    path2 = data_folder+"/avg_vl_"+str(i)+".txt"
    file = open(path2, "w+")
    content = str(avg_valid_loss)
    file.write(content)
    file.close()
    path4 = data_folder+"/el_"+str(i)+".txt"
    file = open(path4, "w+")
    content = str(epoch_loss)
    file.write(content)
    file.close()
    path5 = data_folder+"/vl_"+str(i)+".txt"
    file = open(path5, "w+")
    content = str(valid_loss)
    file.write(content)
    file.close()

end = time.time()
total_time = end-start
print("Total time for training and dse = ", total_time)
print('Min validation loss indexes : ', min_val_index)