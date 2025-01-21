# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


from data import KiwiDataset
from torch import load
from torch.utils.data import ConcatDataset

dataset_list = []
for i in range(11):
    dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}.pt'))
dataset_list.append(load(f'data/kiwi_dataset_1100-1172.pt'))

concatenated_dataset = ConcatDataset(dataset_list)

batch_size = 4

train_set, test_set = torch.utils.data.random_split(concatenated_dataset, [1100, 72])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
manifold = PoincareBall(c=Curvature(requires_grad=True))


import torch
import torch.nn as nn
import hypll as hnn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = hnn.HConvolution2d(
            in_channels=180, out_channels=10, kernel_size=5, manifold=manifold
        )
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv2 = hnn.HConvolution2d(
            in_channels=10, out_channels=5, kernel_size=5, manifold=manifold
        )
        self.fc1 = hnn.HLinear(in_features=10080, out_features=128, manifold=manifold)
        self.fc2 = hnn.HLinear(in_features=128, out_features=64, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=64, out_features=1, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.flatten()
        return x


net = Net()


from hypll.optim import RiemannianAdam


criterion = nn.MSELoss()
optimizer = RiemannianAdam(net.parameters(), lr=0.001, momentum=0.9)



from hypll.tensors import TangentTensor

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # move the inputs to the manifold
        tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
        manifold_inputs = manifold.expmap(tangents)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(manifold_inputs)
        loss = criterion(outputs.tensor, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

PATH = './models/test_convnet_hyper.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

total_loss = 0.
n_examples = 0
# since we're not training, we don't need to calculate the gradients for our outputs
all_labels = []
predicted_labels = []
with torch.no_grad():
    for data in testloader:
        inputs, labels = data

        # move the inputs to the manifold
        tangents = TangentTensor(data=inputs, man_dim=1, manifold=manifold)
        manifold_inputs = manifold.expmap(tangents)

        # calculate outputs by running images through the network
        outputs = net(manifold_inputs)
        loss = criterion(outputs, labels)
        total_loss += loss
        n_examples += len(labels)
        all_labels += labels.tolist()
        predicted_labels += outputs.tolist()

print(f'Average MSE: {total_loss / n_examples}')

from sklearn.metrics import r2_score

r2 = r2_score(all_labels, predicted_labels)

print(f'R2: {r2}')
