from dataset import ElectricityTrainDataset, ElectricityTestDataset, X_endog_train, X_exog_train, y_endog_train, X_endog_test, X_exog_test, y_endog_test
import torch
from torch.utils.data import Dataset, DataLoader


batch_size = 32
elec_train_dataset = ElectricityTrainDataset(X_endog_train, X_exog_train, y_endog_train)
elec_test_dataset = ElectricityTestDataset(X_endog_test, X_exog_test, y_endog_test)

elec_train_dataloader = DataLoader(elec_train_dataset, batch_size = batch_size, shuffle = True)
elec_test_dataloader = DataLoader(elec_test_dataset, batch_size = 64, shuffle = True)

del X_endog_train, X_exog_train, y_endog_train, X_endog_test, X_exog_test, y_endog_test

