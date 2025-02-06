# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch

from data import KiwiDataset
from load_data import load_dataset
from load_data import load_dummy_dataset, load_median_dataset
from models import get_model

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt


N_BINS=20


def train(args):
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    ## DATALOADERS ##
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
    else:
        dataset = load_dataset(label_type=args.dataset_label_type, classification=args.classification, n_bins=N_BINS)
        train_size = 1100
        test_size = 72
        n_classes = N_BINS


    print(f'dataset size {len(dataset)}')
    batch_size = 4

    if args.set_data_split:
        train_set = torch.utils.data.Subset(dataset, range(train_size))
        test_set = torch.utils.data.Subset(dataset, range(train_size, train_size+test_size))
    else:
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

    shuffle = not args.set_data_split
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=2)

    ## TRAIN ## 
    if args.classification:
        pathtmp = "classif"
    else:
        pathtmp="regress"
    
    if args.resnet:
        pathtmp2="resnet"
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
    
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        print("Starting Training!")

        nan_ctr=0

        for epoch in range(args.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                if torch.isnan(outputs).any():
                    print(f'output is a nan yo, {torch.isnan(inputs).sum()} NaNs, iteration {i}')
                    print(outputs)
                    break


                if args.classification: labels = labels.long()

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

        print(f'{nan_ctr} NaNs')
        print('Finished Training')

        torch.save(net.state_dict(), model_path)
        print('saved to', model_path)

    if args.eval_only:
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

    args = parser.parse_args()
    print(args)

    train(args)


if __name__ == "__main__":
    main()
