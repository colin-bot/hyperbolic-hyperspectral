# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


from data import KiwiDataset
from torch import load
from torch.utils.data import ConcatDataset

dataset_list = []
for i in range(11):
    dataset_list.append(load(f'data/kiwi_dataset_{i*100}-{(i+1)*100}'))

concatenated_dataset = ConcatDataset(dataset_list)

batch_size = 4

train_set, test_set = torch.utils.data.random_split(concatenated_dataset, [16, 4])

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
            in_channels=3, out_channels=6, kernel_size=5, manifold=manifold
        )
        self.pool = hnn.HMaxPool2d(kernel_size=2, manifold=manifold, stride=2)
        self.conv2 = hnn.HConvolution2d(
            in_channels=6, out_channels=16, kernel_size=5, manifold=manifold
        )
        self.fc1 = hnn.HLinear(in_features=16 * 5 * 5, out_features=120, manifold=manifold)
        self.fc2 = hnn.HLinear(in_features=120, out_features=84, manifold=manifold)
        self.fc3 = hnn.HLinear(in_features=84, out_features=10, manifold=manifold)
        self.relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


from hypll.optim import RiemannianAdam


criterion = nn.CrossEntropyLoss()
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
