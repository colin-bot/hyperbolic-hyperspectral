from load_data import get_dataset
from data import KiwiDataset
import torch
import argparse
import numpy as np
import random
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def train_plsr(args):
    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    dataset, train_size, val_size, test_size, n_classes = get_dataset(args)
    
    generator1 = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator1)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_size)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1)
    X = next(iter(trainloader))[0].flatten(start_dim=1).numpy()
    y = next(iter(trainloader))[1].numpy()
    print(X.shape, y.shape)

    if args.baseline_type == 'plsr':
        model = PLSRegression(n_components=5)
    elif args.baseline_type == 'svr':
        model = SVR(C=1.0e7, gamma=1.0e4)
    elif args.baseline_type == 'linear':
        model = LinearRegression()
    
    model.fit(X, y)

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
        preds = model.predict(inputs)
        if isinstance(preds, np.ndarray):
            preds = preds[0]
        predicted_y.append(preds)
        true_y_hist.append(int(np.digitize(labels, bin_edges[1:-1])[0]))
        predicted_y_hist.append(int(np.digitize(preds, bin_edges[1:-1])))

    print(len(true_y))
    print(len(np.unique(np.array(predicted_y))))

    r2 = r2_score(true_y, predicted_y)
    print(f'R2: {r2}')

    rmse = np.sqrt(((np.array(true_y) - np.array(predicted_y)) ** 2).mean())
    print(f'RMSE: {rmse}')

    bias = (np.array(true_y) - np.array(predicted_y)).mean()
    sep = np.sqrt(1/(len(true_y) - 1) * ((np.array(true_y) - np.array(predicted_y) - bias) ** 2).sum())
    rpd = np.std(true_y) / sep
    print(f'RPD: {rpd}')

    print(bin_edges[1:-1])
    print(true_y[:10])
    print(predicted_y[:10])
    print(true_y_hist[:10])
    print(predicted_y_hist[:10])

    correct = sum(1 for i, j in zip(true_y_hist, predicted_y_hist) if i == j)
    print(correct)
    
    test_acc = correct / len(true_y_hist)

    print('acc@1=', test_acc)

    save_path = f"{args.baseline_type}_{args.dataset_label_type}_seed{args.seed}"

    if args.plot_preds:
        plt.scatter(true_y, predicted_y)
        min_val, max_val = min(true_y + predicted_y), max(true_y + predicted_y)
        plt.plot(np.arange(min_val, max_val, step=0.1), np.arange(min_val, max_val, step=0.1))
        plt.xlabel(f"true {args.dataset_label_type}")
        plt.ylabel(f"predicted {args.dataset_label_type}")
        plt.title(f"R={r2}, RMSE={rmse}, RPD={rpd}")
        plt.savefig(f"./imgs/{save_path}.png")
        plt.show()

    file = open("output.txt", "a")
    file.write(f"{save_path}, r2 {r2}, rmse {rmse}, rpd {rpd}, test_acc {test_acc}\n")
    file.close()


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
    parser.add_argument("--combined_loss", action='store_true') #dummy
    parser.add_argument("--pca_components", type=int, default=0) # dim reduction
    parser.add_argument("--baseline_type", type=str, default='plsr') #dummy

    args = parser.parse_args()
    print(args)

    train_plsr(args)



if __name__ == "__main__":
    main()
