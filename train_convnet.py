# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch

from data import KiwiDataset
from load_data import load_dataset

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt


N_BINS=20


class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(180, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 5, 5)
        self.fc1 = nn.Linear(10080, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.flatten(x) # for MSELoss
        return x


class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(180, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 5, 5)
        self.fc1 = nn.Linear(10080, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, N_BINS)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(args):
    ## DATALOADERS ##
    dataset = load_dataset(label_type=args.dataset_label_type, classification=args.classification, n_bins=N_BINS)

    batch_size = 4

    train_size = 1100
    test_size = 72

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
    PATH = f"{pathtmp}_{args.dataset_label_type}_convnet_{args.n_epochs}eps"
    model_path = f'./models/{PATH}.pth'
    
    if args.classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if not args.eval_only:
        if args.classification:
            net = ClassificationNet()
        else:
            net = RegressionNet()
    
        optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

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
                    print(outputs)
                    print(torch.isnan(inputs).any())
                    print(torch.isnan(inputs).sum())
                    print('output is a nan yo')
                    print(i)
                    break


                if args.classification: labels = labels.long()

                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print(loss)
                    print(labels)
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
        if args.classification:
            net = ClassificationNet()
        else:
            net = RegressionNet()
        net.load_state_dict(torch.load(model_path, weights_only=True))
        print('loaded from', model_path)

    ## EVAL ##
    total_loss = 0.
    total_correct = 0
    n_examples = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
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
        print(f'Accuracy: {total_correct/n_examples}')
    else:
        print(f'Average MSE: {total_loss / n_examples}')
        r2 = r2_score(all_labels, predicted_labels)
        print(f'R2: {r2}')
        print(all_labels[:10], predicted_labels[:10])

    if args.plot_preds:
        plt.scatter(all_labels, predicted_labels)
        plt.xlabel(f"true {args.dataset_label_type}")
        plt.ylabel(f"predicted {args.dataset_label_type}")
        plt.savefig(f"./imgs/{PATH}.png")
        plt.show()


def main():
    parser=argparse.ArgumentParser(description="Argparser for baseline training script") 
    parser.add_argument("--dataset_label_type", type=str, default="brix")
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--eval_only", action='store_true')
    parser.add_argument("--plot_preds", action='store_true')
    parser.add_argument("--classification", action='store_true')
    parser.add_argument("--set_data_split", action='store_true')
    
    args = parser.parse_args()
    print(args)

    train(args)


if __name__ == "__main__":
    main()
