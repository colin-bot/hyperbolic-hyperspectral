from data import KiwiDataset
from torch.utils.data import ConcatDataset
import torch
import numpy as np


def z_score(tens):
    if torch.isnan(tens).any(): print('original data contains NaN!')
    return (tens - torch.mean(tens, dim=3, keepdim=True)) / torch.std(tens, dim=3, keepdim=True)


def load_dataset(label_type='brix', classification=False, n_bins=20):
    if label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    if classification:
        _, bin_edges = np.histogram(labels_arr, bins=n_bins)
        labels_arr = np.digitize(labels_arr, bin_edges[1:-1])

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = torch.load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt')
        dataset_tmp.samples = z_score(dataset_tmp.samples)
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = torch.load(f'data/kiwi_dataset_1100-1172.pt')
    dataset_tmp.samples = z_score(dataset_tmp.samples)
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def main(): # for testing
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


if __name__ == "__main__":
    main()
