from data import KiwiDataset
from torch import load
from torch.utils.data import ConcatDataset
import torch
from numpy import load as npload

dataset_list = []
for i in range(11):
    dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))

concatenated_dataset = ConcatDataset(dataset_list)

trainloader = torch.utils.data.DataLoader(concatenated_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)

concatenated_dataset.labels = torch.tensor(npload('data/penetros.npy'))

trainloader = torch.utils.data.DataLoader(concatenated_dataset, batch_size=4,
                                          shuffle=True, num_workers=2)

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)
