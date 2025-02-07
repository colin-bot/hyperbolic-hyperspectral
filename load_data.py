from data import KiwiDataset
from torch.utils.data import ConcatDataset
import torch
import numpy as np


def z_score(tens):
    if torch.isnan(tens).any(): print('original data contains NaN!')
    # print(torch.mean(torch.std(tens, dim=3, keepdim=True)))
    tens_norm = (tens - torch.mean(tens, dim=3, keepdim=True)) / torch.std(tens, dim=3, keepdim=True).clamp(0.0001)
    if torch.isnan(tens_norm).any(): print('normalized data contains NaN!')
    return tens_norm.permute(0,3,1,2)


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


def load_dummy_dataset():
    # classification dataset thats either kiwi data or random
    dataset_list = []

    dataset_tmp = torch.load(f'data/kiwi_dataset_0-100.pt')
    dataset_tmp.samples = z_score(dataset_tmp.samples)
    dataset_tmp.labels = torch.ones(len(dataset_tmp.labels))
    dataset_list.append(dataset_tmp)
    dataset_tmp = torch.load(f'data/kiwi_dataset_100-200.pt')
    dataset_tmp.samples = torch.normal(mean=torch.zeros_like(dataset_tmp.samples), std=1.0)
    dataset_tmp.labels = torch.zeros(len(dataset_tmp.labels))
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def load_median_dataset(label_type='brix'):
    if label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    median = np.median(labels_arr)
    labels_arr[labels_arr < median] = 0
    labels_arr[labels_arr >= median] = 1

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


def load_extremes_dataset(label_type='brix', quantile=0.25):
    if label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    lower_extremes = np.quantile(labels_arr, quantile)
    upper_extremes = np.quantile(labels_arr, (1-quantile))

    labels_arr2 = np.zeros_like(labels_arr)
    labels_arr2[labels_arr < lower_extremes] = 1
    labels_arr2[labels_arr > upper_extremes] = 1

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = torch.load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt')
        dataset_tmp.samples = z_score(dataset_tmp.samples)
        dataset_tmp.labels = torch.tensor(labels_arr2[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = torch.load(f'data/kiwi_dataset_1100-1172.pt')
    dataset_tmp.samples = z_score(dataset_tmp.samples)
    dataset_tmp.labels = torch.tensor(labels_arr2[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def get_dataset(args):
    if args.dataset_label_type == "dummy":
        dataset = load_dummy_dataset()
        train_size = 180
        test_size = 20
        n_classes = 2
    elif "median" in args.dataset_label_type:
        print(args.dataset_label_type)
        dataset = load_median_dataset(label_type=args.dataset_label_type.split('_')[1])
        train_size = 1100
        test_size = 72
        n_classes = 2
    elif "extremes" in args.dataset_label_type:
        split = args.dataset_label_type.split('_')
        dataset = load_extremes_dataset(label_type=split[1], quantile=float(split[2]))
        train_size = 1100
        test_size = 72
        n_classes = 2
    else:
        dataset = load_dataset(label_type=args.dataset_label_type, classification=args.classification, n_bins=args.n_bins)
        train_size = 1100
        test_size = 72
        n_classes = args.n_bins
    
    return dataset, train_size, test_size, n_classes


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
