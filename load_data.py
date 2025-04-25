from data import KiwiDataset, WrapperDataset, Random90DegRot
from torch.utils.data import ConcatDataset
import torch
import numpy as np
from skimage.measure import block_reduce


def spectral_pooling(tens, factor=4, method='avg'):
    if method == 'avg':
        func = np.mean
    elif method == 'max':
        func = np.max
    elif method == 'min':
        func = np.min
    # ugly conversion between tensor/np, needed for skimage block_reduce
    tmp = block_reduce(tens.numpy(), block_size=(1,1,1,factor), func=func)
    new_tens = torch.from_numpy(tmp)
    return new_tens


def z_score(tens):
    if torch.isnan(tens).any(): print('original data contains NaN!')
    # print(torch.mean(torch.std(tens, dim=3, keepdim=True)))
    tens_norm = (tens - torch.mean(tens, dim=3, keepdim=True)) / torch.std(tens, dim=3, keepdim=True).clamp(0.0001)
    if torch.isnan(tens_norm).any(): print('normalized data contains NaN!')
    return tens_norm 


def load_wrap_normalize(filepath, args):
    dataset_tmp = torch.load(filepath)
    dataset_tmp = WrapperDataset(dataset_tmp, transform=None)
    # dataset_tmp = WrapperDataset(dataset_tmp, transform=Random90DegRot(dims=[1,2]))
    if args.pooling_factor != 1: 
        samples_tmp = spectral_pooling(dataset_tmp.samples, factor=args.pooling_factor, method=args.pooling_func)
    else: samples_tmp = dataset_tmp.samples
    samples_tmp = z_score(samples_tmp)
    samples_tmp = samples_tmp.permute(0, 3, 1, 2) # B,X,Y,C -> B,C,X,Y
    dataset_tmp.samples = samples_tmp
    return dataset_tmp


def load_dataset(args):
    if args.dataset_label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if args.dataset_label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if args.dataset_label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    if args.classification:
        _, bin_edges = np.histogram(labels_arr, bins=args.n_bins)
        labels_arr = np.digitize(labels_arr, bin_edges[1:-1])

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt', args)
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_1100-1172.pt', args)
    print(dataset_tmp.samples.size())
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def load_dataset_combined(args):
    if args.dataset_label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if args.dataset_label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if args.dataset_label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    _, bin_edges = np.histogram(labels_arr, bins=args.n_bins)
    labels_clf_arr = np.digitize(labels_arr, bin_edges[1:-1])

    labels_arr = [[float(x[0]), float(x[1])] for x in zip(labels_arr, labels_clf_arr)]

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt', args)
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_1100-1172.pt', args)
    print(dataset_tmp.samples.size())
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, bin_edges


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


def load_median_dataset(args):
    label_type=args.dataset_label_type.split('_')[1]
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
        dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt', args)
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_1100-1172.pt', args)
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def load_extremes_dataset(args):
    split = args.dataset_label_type.split('_')
    label_type=split[1]
    quantile=float(split[2])
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
        dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt', args)
        dataset_tmp.labels = torch.tensor(labels_arr2[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_1100-1172.pt', args)
    dataset_tmp.labels = torch.tensor(labels_arr2[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def load_easy_dataset(args):
    label_type = args.dataset_label_type.split('_')[1]
    if label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    labels_arr2 = np.zeros_like(labels_arr)
    if label_type=='brix':
        tasty_threshold=13
        labels_arr2[labels_arr < tasty_threshold] = 0
        labels_arr2[labels_arr >= tasty_threshold] = 1
    else:
        soft_threshold=5
        firm_threshold=11
        labels_arr2[labels_arr < soft_threshold] = 1
        labels_arr2[labels_arr > firm_threshold] = 2

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt', args)
        dataset_tmp.labels = torch.tensor(labels_arr2[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load_wrap_normalize(f'data/kiwi_dataset_1100-1172.pt', args)
    dataset_tmp.labels = torch.tensor(labels_arr2[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def get_dataset(args):
    if args.dataset_label_type == "dummy":
        dataset = load_dummy_dataset()
        train_size = 160
        val_size = 20
        test_size = 20
        n_classes = 2
    elif "median" in args.dataset_label_type:
        print(args.dataset_label_type)
        dataset = load_median_dataset(args)
        train_size = 1000
        val_size = 72
        test_size = 100
        n_classes = 2
    elif "extremes" in args.dataset_label_type:
        dataset = load_extremes_dataset(args)
        train_size = 1000
        val_size = 72
        test_size = 100
        n_classes = 2
    elif "easy" in args.dataset_label_type:
        dataset = load_easy_dataset(args)
        train_size = 1000
        val_size = 72
        test_size = 100
        if args.dataset_label_type=='brix':
            n_classes = 2
        else:
            n_classes = 3
    else:
        if args.combined_loss:
            dataset = load_dataset_combined(args)
        else:
            dataset = load_dataset(args)
        train_size = 1000
        val_size = 72
        test_size = 100
        n_classes = args.n_bins
    
    return dataset, train_size, val_size, test_size, n_classes


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
