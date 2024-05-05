import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from dataset import X_endog_test, X_exog_test, y_endog_test
from sklearn.metrics import mean_absolute_percentage_error
import torch
import os
import datetime
torch.manual_seed(156)
device = torch.device('cuda')
save_path = "C:/Users/Eddie/Desktop/GlobalConditionalWavenets"
wavenet = torch.load(os.path.join(save_path, 'univariate_GCwavenet.pt'))



def auto_infer(model, X_endog, X_exog, y_endog, input_len, output_len, device):
  model.eval()
  model.to(device)
  samples = X_endog.shape[0]
  idx = np.random.choice(samples - 1)
  full_array = np.zeros([1, 1, X_endog.shape[1] + output_len])
  full_array[:, :, :input_len] = X_endog[idx, :]
  infer_array = np.zeros([output_len])
  gt = y_endog[idx, :]
  for i in range(output_len):

    input_array = full_array[:, :, i:i+input_len]
    input = torch.FloatTensor(input_array).to(device)
    output = model(input)
    full_array[:, :, input_len+i] = output.cpu().detach().numpy()
    infer_array[i] = output.reshape(-1).cpu().detach().numpy()


  return infer_array, gt

inferred = auto_infer(wavenet, X_endog_test, X_exog_test, y_endog_test, 512, 72, device)


def generate_time_index(start_time, length_in_minutes):
  end_time = start_time + datetime.timedelta(minutes=length_in_minutes)

  time_index = []

  # Generate time index with 5-minute intervals
  current_time = start_time
  while current_time < end_time:
    time_index.append(current_time)
    current_time += datetime.timedelta(minutes=5)

  return time_index

def plot_inferred(infer_array, gt):
  infer_array = infer_array.reshape(-1)
  gt = gt.reshape(-1)
  mape = mean_absolute_percentage_error(infer_array, gt)
  title = "mape: ", str(mape)
  time_idx = generate_time_index(start_time = datetime.datetime(2020, 1, 1, 9, 0), length_in_minutes = int(5*72))
  fig = plt.figure(figsize=(8, 8))
  fig.set_facecolor('white')
  ax = fig.add_subplot()

  ax.plot(time_idx, infer_array, marker='o', label='pred')
  ax.plot(time_idx, gt, marker='v', label='target')
  plt.title(title)

  plt.show()





