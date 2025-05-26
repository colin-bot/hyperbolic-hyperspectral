# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch

from data import KiwiDataset, Random90DegRot
from load_data import get_dataset
# from models import get_model

import os
import sys

# working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
# os.chdir(working_dir)

# lib_path = os.path.join(working_dir)
# sys.path.append(lib_path)

from classification.utils.initialize import select_dataset, select_model, select_optimizer, load_checkpoint
from torch.nn import DataParallel

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt

from hypll.optim import RiemannianAdam
from scipy.ndimage import gaussian_filter1d


def transform_inputs(inputs, data_transforms, special_modes):
    if 'center_crop' in data_transforms:
        inputs = inputs[:,:,80:100,80:100]
    elif 'avg1d' in special_modes:
        inputs = inputs.mean(dim=(2,3))
    return inputs


def blur_labels(labels, num_classes, device):
    labels = F.one_hot(labels.cpu(), num_classes=num_classes).float()
    labels = torch.tensor(gaussian_filter1d(labels, sigma=1, axis=1)).to(device)
    return labels


class CombinedLoss(nn.Module):
    def __init__(self, bin_edges, weights=(0.01,1.,0.1), regularization_mode='l1', blur_labels=False, num_classes=8, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.bin_edges = bin_edges
        self.weights = weights
        self.mse = nn.MSELoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.regularization_mode = regularization_mode
        self.blur_labels = blur_labels
        self.num_classes = num_classes
        self.device = device
    
    def forward(self, predictions, targets):
        regr_preds = predictions[:,0]            
        clf_preds = predictions[:, 1:]
        targets = targets.flatten()
        clf_targets = targets[1::2].long()
        regr_targets = targets[::2]

        regr_loss = self.mse(regr_preds, regr_targets)
        if self.blur_labels:
            clf_targets = blur_labels(clf_targets, num_classes=self.num_classes, device=self.device)
        clf_loss = self.crossentropy(clf_preds, clf_targets)
        regularization_loss = self.regularization_term(regr_preds, clf_preds)

        loss = regr_loss * self.weights[0] + clf_loss * self.weights[1] + regularization_loss * self.weights[2]
        
        return loss

    def regularization_term(self, regr_preds, clf_preds):
        predicted_bins = torch.max(clf_preds, dim=1).indices
        predicted_bin_centers = (self.bin_edges[predicted_bins] + self.bin_edges[predicted_bins + 1]) / 2
        if self.regularization_mode == 'l1':
            loss = torch.mean((predicted_bin_centers - regr_preds).abs())
        elif self.regularization_mode == 'l2':
            loss = torch.mean((predicted_bin_centers - regr_preds) ** 2)
        return loss


class ModelArgs:
    def __init__(self,
                 classification=True,
                 resnet=True,
                 special_modes=None,
                 hypll=False,
                 pooling_factor=1,
                 pooling_func='avg',
                 onebyoneconv=False,
                 onebyoneconvdim=1,
                 combined_loss=False):
        self.classification = classification
        self.resnet = resnet
        self.special_modes = special_modes
        self.hypll = hypll
        self.pooling_factor = pooling_factor
        self.pooling_func = pooling_func
        self.onebyoneconv = onebyoneconv
        self.onebyoneconvdim = onebyoneconvdim
        self.combined_loss=combined_loss


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.seed != 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    ## DATALOADERS ##
    dataset, train_size, val_size, test_size, n_classes = get_dataset(args)
    if args.combined_loss:
        bin_edges = dataset[1]
        dataset = dataset[0]
        print(bin_edges)
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

    # if args.classification:
    #     pathtmp = "classif"
    # else:
    #     pathtmp="regress"
    
    # if args.hypll:
    #     pathtmp2="poincare"
    # elif args.resnet:
    #     pathtmp2="resnet"
    # elif 'avg1d' in special_modes:
    #     pathtmp2="avg1d"
    # else:
    #     pathtmp2="convnet"
    
    # save_path_euc = f"{pathtmp}_{args.dataset_label_type}_{pathtmp2}_{args.n_epochs}eps_seed{args.seed}"
    # model_path_euc = f'./models/{save_path}.pth'

    # euc_args = ModelArgs(classification=args.classification, 
    #                      resnet=True, 
    #                      special_modes=args.special_modes,
    #                      hypll=False,
    #                      pooling_factor=args.pooling_factor,
    #                      pooling_func=args.pooling_func,
    #                      onebyoneconv=args.onebyoneconv,
    #                      onebyoneconvdim=args.onebyoneconvdim,
    #                      combined_loss=False)
    # hyp_args = ModelArgs(classification=args.classification, 
                        #  resnet=False, 
                        #  special_modes=args.special_modes,
                        #  hypll=True,
                        #  pooling_factor=args.pooling_factor,
                        #  pooling_func=args.pooling_func,
                        #  onebyoneconv=args.onebyoneconv,
                        #  onebyoneconvdim=args.onebyoneconvdim)

    # model_path_euc = f'{working_dir}/classification/good_models/classif_{args.dataset_label_type}_resnet_30eps_seed{args.seed}.pth'
    # net_euc = get_model(euc_args, n_classes=n_classes).to(device)
    # net_euc.load_state_dict(torch.load(model_path_euc, weights_only=False))
    # print('loaded from', model_path_euc)

    # model_path_hyp = f'./good_models/classif_{args.dataset_label_type}_poincare_5eps_seed{args.seed}.pth'
    # net_hyp = get_model(hyp_args, n_classes=n_classes).to(device)
    # net_hyp.load_state_dict(torch.load(model_path_hyp, weights_only=False))
    # print('loaded from', model_path_hyp)

    img_dim = [204//args.pooling_factor, 180, 180]

    if args.combined_loss:
        num_classes = 1 + args.n_bins
    elif args.classification:
        num_classes = args.n_bins
    else:
        num_classes = 1

    net = select_model(img_dim, num_classes, args).to(device)
    device_tmp = [device + ':0']
    net = DataParallel(net, device_ids=device_tmp)
    # model_path_hyp = f'{working_dir}/classification/good_models/best_L-ResNet18-brix.pth'
    # checkpoint = torch.load(model_path_hyp, map_location=device)
    # net_hyp.module.load_state_dict(checkpoint['model'], strict=True)

    n_params = 0
    for name, param in net.named_parameters():
        n_params += param.numel()
    print(f'{n_params} total parameters')

    ## EVAL ##
    if args.combined_loss:
        criterion = CombinedLoss(bin_edges=torch.tensor(bin_edges).to(device), blur_labels=args.blur_labels, num_classes=n_classes, device=device)
    elif args.classification:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.MSELoss()

    total_loss = 0.
    total_correct = 0
    n_examples = 0
    all_labels = []
    predicted_labels = []

    hyp_weight = args.hyp_weight

    model_path = f'models/hypcv_{args.seed}.pt'

    if not args.eval_only:
        # net = get_model(args, n_classes=n_classes).to(device)
    
        # print(net)

        optimizer, lr_scheduler = select_optimizer(net, args)

        print("Starting Training!")

        nan_ctr=0
        best_val_loss=np.inf
        best_val_acc = -1.
        early_stopping_ctr=0

        if 'rd90rot' in data_transforms:
            augmentation = Random90DegRot(dims=[2,3])
        else:
            augmentation = None

        for epoch in range(args.n_epochs):
            # TRAIN
            net.train()
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0], data[1]
                if augmentation:
                    inputs = augmentation(inputs)

                inputs = transform_inputs(inputs, data_transforms, special_modes)

                if not args.combined_loss and args.classification: labels = labels.long()
                else: labels = labels.flatten()

                if not args.combined_loss and args.blur_labels:
                    labels = blur_labels(labels, num_classes=n_classes, device=device)

                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                if args.hypll: outputs = outputs.tensor

                if torch.isnan(outputs).any():
                    print(f'output is a nan yo, {torch.isnan(inputs).sum()} NaNs, iteration {i}')
                    print(outputs)
                    break

                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print(f'loss {loss} is a nan at iter {i}, labels: {labels}')
                    nan_ctr += 1

                loss.backward()


                # break

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print(f'minibatch loss {loss.item()}')
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                
                # print(loss.item())
            
            # VALIDATION
            eval_every_n_epochs = 1 #todo make into args?
            early_stopping_threshold = 5

            if epoch % eval_every_n_epochs == 0:
                net.eval()
                val_loss = 0.
                true_labels = []
                predicted_labels = []
                correct = 0
                first_minibatch = True
                with torch.no_grad():
                    for data in valloader:
                        inputs, labels = data[0].to(device), data[1].to(device)
                        inputs = transform_inputs(inputs, data_transforms, special_modes)
                        # calculate outputs by running images through the network
                        outputs = net(inputs)
                        if args.hypll: outputs = outputs.tensor
                        if not args.combined_loss and args.classification: labels = labels.long()
                        loss = criterion(outputs, labels)
                        val_loss += loss
    
                        if args.combined_loss:
                            clf_labels = labels.flatten()[1::2]
                            true_labels += clf_labels.tolist()
                            _, predicted = torch.max(outputs[:, 1:], 1)
                            correct += (predicted == clf_labels).sum().item()
                        else:
                            true_labels += labels.tolist()
                            _, predicted = torch.max(outputs, 1)
                            correct += (predicted == labels).sum().item()

                        predicted_labels += predicted.tolist()

                        if first_minibatch:
                            print(labels, predicted)
                            print(outputs)
                            first_minibatch = False
            
                val_acc = correct / len(true_labels)
                if args.classification and not args.combined_loss:
                    if val_acc > best_val_acc:
                        print(f'New best validation accuracy: {val_acc}')
                        best_val_acc = val_acc
                        torch.save(net.state_dict(), model_path)
                        print('saved to', model_path)
                        early_stopping_ctr = 0
                    else:
                        early_stopping_ctr += 1
                        if early_stopping_ctr >= early_stopping_threshold:
                            print(f'{early_stopping_threshold} consecutive validation epochs with worse accuracy, stopping training.')
                            break
                else: 
                    if val_loss < best_val_loss:
                        print(f'New best validation loss: {val_loss}')
                        if args.combined_loss: print(f'val acc {val_acc}')
                        best_val_loss = val_loss
                        torch.save(net.state_dict(), model_path)
                        print('saved to', model_path)
                        early_stopping_ctr = 0
                    else:
                        early_stopping_ctr += 1
                        if early_stopping_ctr >= early_stopping_threshold:
                            print(f'{early_stopping_threshold} consecutive validation epochs with worse loss, stopping training.')
                            break
                

    # net_euc.eval()
    net.eval()
    total_loss = 0.
    total_correct = 0
    n_examples = 0
    all_labels = []
    predicted_labels = []
    regr_labels = []
    regr_preds = []

    save_path = f'lorentz_{args.dataset_label_type}_seed_{args.seed}'


    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = transform_inputs(inputs, data_transforms, special_modes)

            # calculate outputs by running images through the network
            outputs = net(inputs)
            if args.classification and not args.combined_loss: labels = labels.long()
            if args.hypll: outputs = outputs.tensor
            loss = criterion(outputs, labels)
            total_loss += loss
            n_examples += len(labels)
            if args.combined_loss:
                tmp_labels = labels.flatten()
                clf_labels = tmp_labels[1::2].long()
                all_labels += clf_labels.tolist()
                regr_labels += tmp_labels[::2].tolist()
                _, predicted = torch.max(outputs[:, 1:], 1)
                regr_preds += outputs[:, 0].flatten().tolist()
                total_correct += (predicted == clf_labels).sum().item()
                predicted_labels += predicted.tolist()
            elif args.classification:
                all_labels += labels.tolist()
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                predicted_labels += predicted.tolist()
            else:
                all_labels += labels.tolist()
                predicted_labels += outputs.tolist()

    if args.combined_loss:
        print(regr_preds[:50])
        print(regr_labels[:50])
        test_acc = total_correct / n_examples
        print(f'Accuracy: {test_acc}')
        r2 = r2_score(regr_labels, regr_preds)
        print(f'R2: {r2}')
        rmse = np.sqrt(((np.array(regr_labels) - np.array(regr_preds)) ** 2).mean())
        print(f'RMSE: {rmse}')
    elif args.classification:
        test_acc = total_correct / n_examples
        print(f'Accuracy: {test_acc}')
        file = open("output.txt", "a")
        file.write(f"{save_path}, test acc {test_acc}\n")
        file.close()
    else:
        print(f'Average MSE: {total_loss / n_examples}')
        r2 = r2_score(all_labels, predicted_labels)
        print(f'R2: {r2}')

    print('ground truth')
    print(all_labels[:50])
    print('predicted')
    print(predicted_labels[:50])
    if args.classification:
        for i in np.unique(all_labels):
            print(i, predicted_labels.count(i), all_labels.count(i))
        label_difference = np.abs(np.array(predicted_labels)-np.array(all_labels))
        print(label_difference[:50])
        print(np.mean(label_difference))

    if args.plot_preds:
        if args.combined_loss: 
            plt.scatter(regr_labels, regr_preds)
            min_val, max_val = min(regr_labels + regr_preds), max(regr_labels + regr_preds)
            plt.title(f"R={r2}, RMSE={rmse}")
        else: 
            plt.scatter(all_labels, predicted_labels)
            min_val, max_val = min(all_labels + predicted_labels), max(all_labels + predicted_labels)

        plt.plot(np.arange(min_val, max_val, step=0.1), np.arange(min_val, max_val, step=0.1))
        plt.xlabel(f"true {args.dataset_label_type}")
        plt.ylabel(f"predicted {args.dataset_label_type}")
        plt.savefig(f'imgs/hypcv_{args.seed}.png')
        plt.show()

    # with torch.no_grad():
    #     for data in testloader:
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         inputs = transform_inputs(inputs, data_transforms, special_modes)

    #         # calculate outputs by running images through the network
    #         # logits_euc = net_euc(inputs)
    #         logits_hyp = net(inputs)

    #         # print(logits_euc)
    #         # print(logits_hyp)

    #         # logits_euc = logits_euc / logits_euc.sum(dim=1).unsqueeze(dim=1)
    #         # logits_hyp = logits_hyp / logits_hyp.sum(dim=1).unsqueeze(dim=1)

    #         # outputs = (1-hyp_weight) * logits_euc + hyp_weight * logits_hyp
    #         outputs = logits_hyp

    #         # print(outputs)

    #         if args.classification: labels = labels.long()
    #         loss = criterion(outputs, labels)
    #         total_loss += loss
    #         n_examples += len(labels)
    #         all_labels += labels.tolist()
    #         if args.classification:
    #             # outputs = F.softmax(outputs, dim=1)
    #             _, predicted = torch.max(outputs, dim=1)
    #             total_correct += (predicted == labels).sum().item()
    #             predicted_labels += predicted.tolist()
    #         else:
    #             predicted_labels += outputs.tolist()


    # if args.classification:
    #     test_acc = total_correct / n_examples
    #     print(f'Accuracy: {test_acc}')
    #     file = open("output.txt", "a")
    #     file.write(f"{save_path}, test acc {test_acc}\n")
    #     file.close()
    # else:
    #     print(f'Average MSE: {total_loss / n_examples}')
    #     r2 = r2_score(all_labels, predicted_labels)
    #     print(f'R2: {r2}')

    # if args.plot_preds:
    #     print(all_labels[:50])
    #     print(predicted_labels[:50])
    #     if args.classification:
    #         for i in np.unique(all_labels):
    #             print(i, predicted_labels.count(i), all_labels.count(i))
    #         label_difference = np.abs(np.array(predicted_labels)-np.array(all_labels))
    #         print(label_difference[:50])
    #         print(np.mean(label_difference))
    #     plt.scatter(all_labels, predicted_labels)
    #     plt.xlabel(f"true {args.dataset_label_type}")
    #     plt.ylabel(f"predicted {args.dataset_label_type}")
    #     plt.savefig(f"classification/output/imgs/{save_path}.png")
    #     plt.show()


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
    parser.add_argument("--hypll", action='store_true')
    parser.add_argument("--hyp_weight", type=float, default=0.5)
    parser.add_argument("--combined_loss", action='store_true')
    parser.add_argument("--blur_labels", action='store_true')


    # Output settings
    parser.add_argument('--exp_name', default="test", type=str,
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str,
                        help="Path for output files (relative to working directory).")

    # Model selection
    parser.add_argument('--num_layers', default=18, type=int, choices=[10, 18, 34, 50],
                        help="Number of layers in ResNet.")
    parser.add_argument('--embedding_dim', default=512, type=int,
                        help="Dimensionality of classification embedding space (could be expanded by ResNet)")
    parser.add_argument('--encoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz"],
                        help="Select conv model encoder manifold.")
    parser.add_argument('--decoder_manifold', default='lorentz', type=str, choices=["euclidean", "lorentz", "poincare"],
                        help="Select conv model decoder manifold.")

    # Hyperbolic geometry settings
    parser.add_argument('--learn_k', action='store_true',
                        help="Set a learnable curvature of hyperbolic geometry.")
    parser.add_argument('--encoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in backbone (geoopt.K=-1/K).")
    parser.add_argument('--decoder_k', default=1.0, type=float,
                        help="Initial curvature of hyperbolic geometry in decoder (geoopt.K=-1/K).")
    parser.add_argument('--clip_features', default=1.0, type=float,
                        help="Clipping parameter for hybrid HNNs proposed by Guo et al. (2022)")


    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--optimizer', default="RiemannianSGD", type=str,
                        choices=["RiemannianAdam", "RiemannianSGD", "Adam", "SGD"],
                        help="Optimizer for training.")
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help="If learning rate should be reduced after step epochs using a LR scheduler.")
    parser.add_argument('--lr_scheduler_milestones', default=[60, 120, 160], type=int, nargs="+",
                        help="Milestones of LR scheduler.")
    parser.add_argument('--lr_scheduler_gamma', default=0.2, type=float,
                        help="Gamma parameter of LR scheduler.")


    args = parser.parse_args()
    print(args)

    train(args)


if __name__ == "__main__":
    main()
