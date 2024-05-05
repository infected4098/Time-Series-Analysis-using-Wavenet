from dataloader import elec_test_dataloader, elec_train_dataloader
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from uniwave6 import Wavenet
from dataloader import elec_train_dataloader, elec_test_dataloader
import warnings
warnings.filterwarnings("ignore")
device = torch.device('cuda')
torch.manual_seed(156)

model = Wavenet(7, 2, 1, 64, 64).to(device)
train_loader = elec_train_dataloader
test_loader = elec_test_dataloader
interval = 500; total_count = 0
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
"""
for name, param in model.named_parameters():
    # print(f"Layer: {name} - Size: {param.numel()}")
    total_count += param.numel()
print("#Total Parameters of this model is :", total_count)
early_stopping_count = 0
"""

optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(interval, model, device, train_loader, test_loader, epoch, optimizer):
    condition_broken = False
    early_stopping_count = 0
    save_path = "C:/Users/Eddie/Desktop/GlobalConditionalWavenets"
    early_stop_thrs = 4
    loss_history = []
    init_loss = 999999
    val_loss_history = []
    criterion = nn.L1Loss().to(device)
    for i in range(epoch):
        model.train()
        running_loss = 0.0
        for batch_idx, (X_train_endog, X_train_exog, y_train_endog) in tqdm(enumerate(train_loader)):

            X_train_endog = X_train_endog.to(device)
            #X_train_exog = X_train_exog.to(device)
            y_train_endog = y_train_endog.to(device)
            optimizer.zero_grad()
            output = model(X_train_endog)
            #loss = criterion(output, X_train_endog[:, :, -1].unsqueeze(2))
            loss = criterion(output, y_train_endog.unsqueeze(2))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx != 0:

                if batch_idx % interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        i, batch_idx * X_train_endog.size(0), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), running_loss/interval))
                    loss_history.append(running_loss/interval)
                    running_loss = 0.0
                    val_loss = 0.0
                    for k, (X_test_endog, X_test_exog, y_test_endog) in enumerate(test_loader):
                        model.eval()
                        with torch.no_grad():
                            X_test_endog = X_test_endog.to(device)
                            #X_test_exog = X_test_exog.to(device)
                            y_test_endog = y_test_endog.to(device)
                            output = model(X_test_endog)
                            loss_ = criterion(output, y_test_endog.unsqueeze(2))
                            val_loss += loss_.item()
                    print(f'[{i + 1}, {k + 1:5d}] val loss: {val_loss / k:.3f}')
                    val_loss_history.append(val_loss / k)

                    if val_loss < init_loss:

                        torch.save(model, os.path.join(save_path, 'lastmodel.pt'))

                        init_loss = val_loss
                        early_stopping_count = 0
                    else:
                        early_stopping_count += 1
                    if early_stopping_count >= early_stop_thrs:

                        print("Early Stopping, Best epoch is {}".format(i))

                        condition_broken = True
    torch.save(model, os.path.join(save_path, '72_newest.pt'))

    np.save("C:/Users/Eddie/Desktop//GlobalConditionalWavenets/g_once_univariate_gcw_val_loss_history.npy", val_loss_history)
    np.save("C:/Users/Eddie/Desktop//GlobalConditionalWavenets/g_once_univariate_gcw_loss_history.npy", loss_history)

train(500, model, device, train_loader, test_loader, 6 , optimizer)