# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch

from data import KiwiDataset
from torch.utils.data import ConcatDataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt

N_BINS=20


def z_score(tens):
    return (tens - torch.mean(tens, dim=3, keepdim=True)) / torch.std(tens, dim=3, keepdim=True)


def load_dataset(label_type='brix', classification=False):
    if label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
    if label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
    if label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')

    if classification:
        _, bin_edges = np.histogram(labels_arr, bins=N_BINS)
        labels_arr = np.digitize(labels_arr, bin_edges[1:-1])

    dataset_list = []
    for i in range(11):
        # dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
        dataset_tmp = torch.load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt')
        dataset_tmp.samples = z_score(dataset_tmp)
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = torch.load(f'data/kiwi_dataset_1100-1172.pt')
    dataset_tmp.samples = z_score(dataset_tmp)
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


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


def run_baseline(args):
    ## DATALOADERS ##
    dataset = load_dataset(label_type=args.dataset_label_type, classification=args.classification)

    batch_size = 4

    train_set, test_set = torch.utils.data.random_split(dataset, [1100, 72])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

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
    
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(args.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                
                if args.classification: labels = labels.long()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

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
    args = parser.parse_args()
    print(args.eval_only)
    run_baseline(args)

if __name__ == "__main__":
    main()
