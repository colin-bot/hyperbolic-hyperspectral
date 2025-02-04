from data import KiwiDataset
from torch import load
from torch.utils.data import ConcatDataset
import torch
from numpy import load as npload


def z_score(tens):
    return (tens - torch.mean(tens, dim=3, keepdim=True)) / torch.std(tens, dim=3, keepdim=True)


def load_dataset(label_type='brix'):
    if label_type=='brix':
        labels_arr = npload('data/brixes.npy')
    if label_type=='aweta':
        labels_arr = npload('data/awetas.npy')
    if label_type=='penetro':
        labels_arr = npload('data/penetros.npy')

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt')
        dataset_tmp.samples = z_score(dataset_tmp)
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load(f'data/kiwi_dataset_1100-1172.pt')
    dataset_tmp.samples = z_score(dataset_tmp)
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset

print('penetro')
dataset = load_dataset(label_type='penetro')

trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)
    break

print('brix')
dataset = load_dataset(label_type='brix')

trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)
    break

print('aweta')
dataset = load_dataset(label_type='aweta')

trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=False, num_workers=2)

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    print(labels)
    break
