# Script to test a hybrid of a Poincare (based on HypLL) and Euclidean model 
# Usage: python3 hybrid_hypll.py --dataset_label_type [brix/aweta/penetro]
# Furter args explained in the main function
# Enter the paths to trained models in the train function

import torch

from data import KiwiDataset, Random90DegRot
from load_data import get_dataset
from models import get_model

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt

from hypll.optim import RiemannianAdam


def transform_inputs(inputs, data_transforms, special_modes):
    if 'center_crop' in data_transforms:
        inputs = inputs[:,:,80:100,80:100]
    elif 'avg1d' in special_modes:
        inputs = inputs.mean(dim=(2,3))
    return inputs


class ModelArgs:
    def __init__(self,
                 classification=True,
                 resnet=True,
                 special_modes=None,
                 hypll=False,
                 pooling_factor=1,
                 pooling_func='avg',
                 onebyoneconv=False,
                 onebyoneconvdim=1):
        self.classification = classification
        self.resnet = resnet
        self.special_modes = special_modes
        self.hypll = hypll
        self.pooling_factor = pooling_factor
        self.pooling_func = pooling_func
        self.onebyoneconv = onebyoneconv
        self.onebyoneconvdim = onebyoneconvdim


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    ## DATALOADERS ##
    dataset, train_size, val_size, test_size, n_classes = get_dataset(args)
    print(f'dataset size {len(dataset)}')

    batch_size = args.batch_size

    if args.set_data_split:
        train_set = torch.utils.data.Subset(dataset, range(train_size))
        val_set = torch.utils.data.Subset(dataset, range(train_size, train_size+val_size))
        test_set = torch.utils.data.Subset(dataset, range(train_size+val_size, train_size+val_size+test_size))
    else:
        generator1 = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator1)

    print('train val test size', len(train_set), len(val_set), len(test_set))

    shuffle = not args.set_data_split
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=2)

    ## TRAIN ## 
    if args.data_transforms: data_transforms = args.data_transforms.split('-')
    else: data_transforms = []
    if args.special_modes: special_modes = args.special_modes.split('-')
    else: special_modes = []

    euc_args = ModelArgs(classification=args.classification, 
                         resnet=True, 
                         special_modes=args.special_modes,
                         hypll=False,
                         pooling_factor=args.pooling_factor,
                         pooling_func=args.pooling_func,
                         onebyoneconv=args.onebyoneconv,
                         onebyoneconvdim=args.onebyoneconvdim)
    hyp_args = ModelArgs(classification=args.classification, 
                         resnet=False, 
                         special_modes=args.special_modes,
                         hypll=True,
                         pooling_factor=args.pooling_factor,
                         pooling_func=args.pooling_func,
                         onebyoneconv=args.onebyoneconv,
                         onebyoneconvdim=args.onebyoneconvdim)

    model_path_euc = f'./good_models/classif_{args.dataset_label_type}_resnet_30eps_seed{args.seed}.pth'
    net_euc = get_model(euc_args, n_classes=n_classes).to(device)
    net_euc.load_state_dict(torch.load(model_path_euc, weights_only=False))
    print('loaded from', model_path_euc)

    model_path_hyp = f'./good_models/classif_{args.dataset_label_type}_poincare_5eps_seed{args.seed}.pth'
    net_hyp = get_model(hyp_args, n_classes=n_classes).to(device)
    net_hyp.load_state_dict(torch.load(model_path_hyp, weights_only=False))
    print('loaded from', model_path_hyp)

    ## EVAL ##
    if args.classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    total_loss = 0.
    total_correct = 0
    n_examples = 0
    all_labels = []
    predicted_labels = []

    hyp_weight = args.hyp_weight

    net_euc.eval()
    net_hyp.eval()

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = transform_inputs(inputs, data_transforms, special_modes)

            # calculate outputs by running images through the network
            logits_euc = net_euc(inputs)
            logits_hyp = net_hyp(inputs).tensor

            # logits_euc = logits_euc / logits_euc.sum(dim=1).unsqueeze(dim=1)
            # logits_hyp = logits_hyp / logits_hyp.sum(dim=1).unsqueeze(dim=1)

            outputs = (1-hyp_weight) * logits_euc + hyp_weight * logits_hyp

            if args.classification: labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss
            n_examples += len(labels)
            all_labels += labels.tolist()
            if args.classification:
                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()
                predicted_labels += predicted.tolist()
            else:
                predicted_labels += outputs.tolist()

    save_path = f'hybrid_{args.dataset_label_type}_seed_{args.seed}'

    if args.classification:
        test_acc = total_correct / n_examples
        print(f'Accuracy: {test_acc}')
        file = open("output.txt", "a")
        file.write(f"{save_path}, test acc {test_acc}\n")
        file.close()
    else:
        print(f'Average MSE: {total_loss / n_examples}')
        r2 = r2_score(all_labels, predicted_labels)
        print(f'R2: {r2}')

    if args.plot_preds:
        print(all_labels[:50])
        print(predicted_labels[:50])
        if args.classification:
            for i in np.unique(all_labels):
                print(i, predicted_labels.count(i), all_labels.count(i))
            label_difference = np.abs(np.array(predicted_labels)-np.array(all_labels))
            print(label_difference[:50])
            print(np.mean(label_difference))
        plt.scatter(all_labels, predicted_labels)
        plt.xlabel(f"true {args.dataset_label_type}")
        plt.ylabel(f"predicted {args.dataset_label_type}")
        plt.savefig(f"./imgs/{save_path}.png")
        plt.show()


def main():
    parser=argparse.ArgumentParser(description="Argparser for baseline training script") 
    parser.add_argument("--dataset_label_type", type=str, default="brix")
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--eval_only", action='store_true')
    parser.add_argument("--plot_preds", action='store_true')
    parser.add_argument("--classification", action='store_true')
    parser.add_argument("--resnet", action='store_true')
    parser.add_argument("--set_data_split", action='store_true')
    parser.add_argument("--seed", type=int, default=0) # 0 = NO SEED!
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001) # low lr by default!
    parser.add_argument("--n_bins", type=int, default=0) # for bin classification task
    parser.add_argument("--data_transforms", type=str) # data transforms / augmentations
    parser.add_argument("--special_modes", type=str) # special network types
    parser.add_argument("--pooling_factor", type=int, default=1) # dim reduction
    parser.add_argument("--pooling_func", type=str) # dim reduction, options 'avg', 'max', 'min'
    parser.add_argument("--onebyoneconv", action='store_true')
    parser.add_argument("--onebyoneconvdim", type=int, default=32)
    parser.add_argument("--hypll", action='store_true')
    parser.add_argument("--hyp_weight", type=float, default=0.5)

    args = parser.parse_args()
    print(args)

    train(args)


if __name__ == "__main__":
    main()
