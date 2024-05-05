from data import dataset_pipeline, half_df
import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda')
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
X_endog_train, X_exog_train, y_endog_train, X_endog_test, X_exog_test, y_endog_test = dataset_pipeline(half_df, train_size = 255, test_size = 1, step = 11, test_prop = 0.2, split = True, enex = True)
print(X_endog_train.shape, X_exog_train.shape, y_endog_train.shape, X_endog_test.shape, X_exog_test.shape, y_endog_test.shape)
del half_df


class ElectricityTrainDataset(Dataset):
    def __init__(self, X_endog_train, X_exog_train,
                 y_endog_train):
        self.X_endog_train = torch.FloatTensor(X_endog_train).unsqueeze(1).to(device)
        self.X_exog_train = torch.FloatTensor(X_exog_train).permute(0, 2, 1).to(device)
        self.y_endog_train = torch.FloatTensor(y_endog_train).to(device)
    def __len__(self):
        return self.X_endog_train.shape[0]

    def __getitem__(self, idx):
        return self.X_endog_train[idx], self.X_exog_train[idx], self.y_endog_train[idx]

class ElectricityTestDataset(Dataset):
    def __init__(self, X_endog_test,
                 X_exog_test, y_endog_test):
        self.X_endog_test = torch.FloatTensor(X_endog_test).unsqueeze(1).to(device)
        self.X_exog_test = torch.FloatTensor(X_exog_test).permute(0, 2, 1).to(device)
        self.y_endog_test = torch.FloatTensor(y_endog_test).to(device)

    def __len__(self):
        return self.X_endog_test.shape[0]

    def __getitem__(self, idx):
        return self.X_endog_test[idx], self.X_exog_test[idx], self.y_endog_test[idx]

