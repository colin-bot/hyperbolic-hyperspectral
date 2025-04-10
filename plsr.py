from load_data import get_dataset
from data import KiwiDataset
import torch
import argparse
import numpy as np
import random
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def train_plsr(args):
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset, train_size, val_size, test_size, n_classes = get_dataset(args)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_size)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1)
    X = next(iter(trainloader))[0].flatten(start_dim=1).numpy()
    y = next(iter(trainloader))[1].numpy()
    print(X.shape, y.shape)

    plsr = PLSRegression(n_components=1)
    plsr.fit(X, y)

    true_y = []
    predicted_y = []
    true_y_hist = []
    predicted_y_hist = []

    label_type = args.dataset_label_type
    if label_type=='brix':
        labels_arr = np.load('data/brixes.npy')
        n_bins = 10
    if label_type=='aweta':
        labels_arr = np.load('data/awetas.npy')
        n_bins = 8
    if label_type=='penetro':
        labels_arr = np.load('data/penetros.npy')
        n_bins = 8

    _, bin_edges = np.histogram(labels_arr, bins=n_bins)
    labels_arr = np.digitize(labels_arr, bin_edges[1:-1])

    for data in testloader:
        inputs, labels = data
        inputs = inputs.flatten(start_dim=1)
        true_y.append(float(labels))
        preds = plsr.predict(inputs)
        predicted_y.append(preds)
        true_y_hist.append(int(np.digitize(labels, bin_edges[1:-1])[0]))
        predicted_y_hist.append(int(np.digitize(preds, bin_edges[1:-1])))

    print(len(true_y))
    print(len(np.unique(np.array(predicted_y))))

    r2 = r2_score(true_y, predicted_y)
    print(f'R2: {r2}')

    print(bin_edges[1:-1])
    print(true_y[:10])
    print(predicted_y[:10])
    print(true_y_hist[:10])
    print(predicted_y_hist[:10])

    correct = sum(1 for i, j in zip(true_y_hist, predicted_y_hist) if i == j)
    print(correct)
    
    print('acc@1=', correct / len(true_y_hist))

    if args.plot_preds:
        plt.scatter(true_y, predicted_y)
        plt.xlabel(f"true {args.dataset_label_type}")
        plt.ylabel(f"predicted {args.dataset_label_type}")
        plt.savefig(f"./imgs/plsr_{args.dataset_label_type}.png")
        plt.show()


def main():
    parser=argparse.ArgumentParser(description="Argparser for PLSR training script") 
    parser.add_argument("--dataset_label_type", type=str, default="brix")
    parser.add_argument("--plot_preds", action='store_true')
    parser.add_argument("--classification", action='store_true')
    parser.add_argument("--set_data_split", action='store_true')
    parser.add_argument("--seed", type=int, default=0) # 0 = NO SEED!
    parser.add_argument("--n_bins", type=int, default=0) # for bin classification task
    parser.add_argument("--pooling_factor", type=int, default=1) # dim reduction
    parser.add_argument("--pooling_func", type=str) # dim reduction, options 'avg', 'max', 'min'


    args = parser.parse_args()
    print(args)

    train_plsr(args)



if __name__ == "__main__":
    main()
