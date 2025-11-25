import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
from Dataset import WeatherDataset

dataset_file = "/Users/minwoongyoon/Documents/Res/weather.csv"

#Test-Train split on date 
split_date = pd.to_datetime('2023-01-01')

# days in the input sequence 
day_range = 15 
days_in = 14 #MLP input days 

assert day_range > days_in 

#parameters 
learning_rate = 1e-4
epochs = 500
batch_size = 32

dataset_train = WeatherDataset(dataset_file, day_range=day_range, split_date=split_date, train_test ="train")
dataset_test = WeatherDataset(dataset_file, day_range=day_range, split_date=split_date, train_test="test")

print(f'{len(dataset_train)}')
print(f'{len(dataset_test)}')

data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

#plotting 
plt.figure(figsize = (10,5))
plt.title("Max Daily Temp (C)")
plt.plot(dataset_train.dataset.index, dataset_train.dataset.values[:, 1])
plt.plot(dataset_test.dataset.index, dataset_test.dataset.values[:, 1])
plt.legend(["Train", "Test"])
plt.grid(True)
plt.savefig('plot')
plt.close()

#Residual MLP 

