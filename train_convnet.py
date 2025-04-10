# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


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


def transform_inputs(inputs, data_transforms, special_modes):
    if 'center_crop' in data_transforms:
        inputs = inputs[:,:,80:100,80:100]
    elif 'avg1d' in special_modes:
        inputs = inputs.mean(dim=(2,3))
    return inputs


def train(args):
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
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

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

    if args.classification:
        pathtmp = "classif"
    else:
        pathtmp="regress"
    
    if args.resnet:
        pathtmp2="resnet"
    elif 'avg1d' in special_modes:
        pathtmp2="avg1d"
    else:
        pathtmp2="convnet"
    
    save_path = f"{pathtmp}_{args.dataset_label_type}_{pathtmp2}_{args.n_epochs}eps_seed{args.seed}"
    model_path = f'./models/{save_path}.pth'
    
    if args.classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if not args.eval_only:
        net = get_model(args, n_classes=n_classes)
    
        # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

        print("Starting Training!")

        nan_ctr=0
        best_val_loss=np.inf
        early_stopping_ctr=0

        if 'rd90rot' in data_transforms:
            augmentation = Random90DegRot(dims=[2,3])
        else:
            augmentation = None

        for epoch in range(args.n_epochs):
            # TRAIN
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if augmentation:
                    inputs = augmentation(inputs)

                inputs = transform_inputs(inputs, data_transforms, special_modes)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                if torch.isnan(outputs).any():
                    print(f'output is a nan yo, {torch.isnan(inputs).sum()} NaNs, iteration {i}')
                    print(outputs)
                    break

                if args.classification: labels = labels.long()
                else: labels = labels.flatten()
                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print(f'loss {loss} is a nan at iter {i}, labels: {labels}')
                    nan_ctr += 1

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                
            
            # VALIDATION
            eval_every_n_epochs = 1 #todo make into args?
            early_stopping_threshold = 5

            if epoch % eval_every_n_epochs == 0:
                val_loss = 0.
                with torch.no_grad():
                    for data in valloader:
                        inputs, labels = data
                        inputs = transform_inputs(inputs, data_transforms, special_modes)
                        # calculate outputs by running images through the network
                        outputs = net(inputs)
                        if args.classification: labels = labels.long()
                        loss = criterion(outputs, labels)
                        val_loss += loss
                if val_loss < best_val_loss:
                    print(f'New best validation loss: {val_loss}')
                    best_val_loss = val_loss
                    torch.save(net.state_dict(), model_path)
                    print('saved to', model_path)
                    early_stopping_ctr = 0
                else:
                    early_stopping_ctr += 1
                    if early_stopping_ctr >= early_stopping_threshold:
                        print(f'{early_stopping_threshold} consecutive validation epochs with worse loss, stopping training.')
                        break

        print(f'{nan_ctr} NaNs')
        print('Finished Training')

    net = get_model(args, n_classes=n_classes)
    net.load_state_dict(torch.load(model_path, weights_only=True))
    print('loaded from', model_path)

    ## EVAL ##
    total_loss = 0.
    total_correct = 0
    n_examples = 0
    all_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = transform_inputs(inputs, data_transforms, special_modes)

            # calculate outputs by running images through the network
            outputs = net(inputs)
            if args.classification: labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss
            n_examples += len(labels)
            all_labels += labels.tolist()
            if args.classification:
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                predicted_labels += predicted.tolist()
            else:
                predicted_labels += outputs.tolist()

    if args.classification:
        print(f'Accuracy: {total_correct / n_examples}')
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

    args = parser.parse_args()
    print(args)

    train(args)


if __name__ == "__main__":
    main()
