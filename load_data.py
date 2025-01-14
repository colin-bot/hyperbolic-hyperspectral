from data import KiwiDataset
from torch import load
from torch.utils.data import ConcatDataset

dataset1 = load('data/kiwi_dataset.pt')
dataset2 = load('data/kiwi_dataset_2.pt')

concatenated_dataset = ConcatDataset([dataset1, dataset2])

print(len(concatenated_dataset))
