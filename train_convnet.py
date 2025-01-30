# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch

from data import KiwiDataset
from torch import load
from torch.utils.data import ConcatDataset
from numpy import load as npload
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse

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
        dataset_tmp.labels = torch.tensor(labels_arr[i*100:(i+1)*100])
        dataset_list.append(dataset_tmp)
    # dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))
    dataset_tmp = load(f'data/kiwi_dataset_1100-1172.pt')
    dataset_tmp.labels = torch.tensor(labels_arr[1100:1172])
    dataset_list.append(dataset_tmp)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


class Net(nn.Module):
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


def run_baseline(dataset_label_type, n_epochs, eval_only):
    ## DATALOADERS ##
    dataset = load_dataset(label_type=dataset_label_type)

    batch_size = 4

    train_set, test_set = torch.utils.data.random_split(dataset, [1100, 72])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    ## TRAIN ## 
    PATH = f'./models/{dataset_label_type}_convnet_{n_epochs}eps.pth'
    criterion = nn.MSELoss()

    if not eval_only:
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        torch.save(net.state_dict(), PATH)
        print('saved to', PATH)

    if eval_only:
        net = Net()
        net.load_state_dict(torch.load(PATH, weights_only=True))
        print('loaded from', PATH)

    ## EVAL ##
    total_loss = 0.
    n_examples = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    all_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss
            n_examples += len(labels)
            all_labels += labels.tolist()
            predicted_labels += outputs.tolist()

    print(f'Average MSE: {total_loss / n_examples}')
    
    r2 = r2_score(all_labels, predicted_labels)

    print(f'R2: {r2}')

def main():
    parser=argparse.ArgumentParser(description="Argparser for baseline training script") 
    parser.add_argument("--dataset_label_type", type=str, default="brix")
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--eval_only", action='store_true')
    args = parser.parse_args()
    print(args.eval_only)
    run_baseline(args.dataset_label_type, args.n_epochs, args.eval_only)

if __name__ == "__main__":
    main()
