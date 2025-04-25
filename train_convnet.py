# based on PyTorch's example
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch

from data import KiwiDataset, Random90DegRot
from load_data import get_dataset
from models import get_model

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt

from hypll.optim import RiemannianAdam

from pytorch_grad_cam import GradCAM, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST
from pytorch_grad_cam.utils.image import show_cam_on_image

def transform_inputs(inputs, data_transforms, special_modes):
    if 'center_crop' in data_transforms:
        inputs = inputs[:,:,80:100,80:100]
    elif 'avg1d' in special_modes:
        inputs = inputs.mean(dim=(2,3))
    return inputs


class CombinedLoss(nn.Module):
    def __init__(self, bin_edges, weights=(0.1,1.,1.), regularization_mode='l1'):
        super(CombinedLoss, self).__init__()
        self.bin_edges = bin_edges
        self.weights = weights
        self.mse = nn.MSELoss()
        self.crossentropy = nn.CrossEntropyLoss()
        self.regularization_mode = regularization_mode
    
    def forward(self, predictions, targets):
        regr_preds = predictions[:,0]            
        clf_preds = predictions[:, 1:]
        targets = targets.flatten()
        clf_targets = targets[1::2].long()
        regr_targets = targets[::2]

        regr_loss = self.mse(regr_preds, regr_targets)
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

    if args.combined_loss:
        pathtmp = "combined"
    elif args.classification:
        pathtmp = "classif"
    else:
        pathtmp="regress"
    
    if args.hypll:
        pathtmp2="poincare"
    elif args.resnet:
        pathtmp2="resnet"
    elif 'avg1d' in special_modes:
        pathtmp2="avg1d"
    else:
        pathtmp2="convnet"
    
    save_path = f"{pathtmp}_{args.dataset_label_type}_{pathtmp2}_{args.n_epochs}eps_seed{args.seed}"
    model_path = f'./models/{save_path}.pth'
    
    if args.combined_loss:
        criterion = CombinedLoss(bin_edges=torch.tensor(bin_edges).to(device))
    else:
        if args.classification:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

    if not args.eval_only:
        net = get_model(args, n_classes=n_classes).to(device)
    
        # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        if args.hypll:
            optimizer = RiemannianAdam(net.parameters(), lr=args.lr)
        else:
            optimizer = optim.Adam(net.parameters(), lr=args.lr)

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
                inputs, labels = data[0].to(device), data[1].to(device)
                if augmentation:
                    inputs = augmentation(inputs)

                inputs = transform_inputs(inputs, data_transforms, special_modes)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                if args.hypll: outputs = outputs.tensor

                if torch.isnan(outputs).any():
                    print(f'output is a nan yo, {torch.isnan(inputs).sum()} NaNs, iteration {i}')
                    print(outputs)
                    break

                if not args.combined_loss and args.classification: labels = labels.long()
                else: labels = labels.flatten()

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
                
            
            # VALIDATION
            eval_every_n_epochs = 1 #todo make into args?
            early_stopping_threshold = 5

            if epoch % eval_every_n_epochs == 0:
                net.eval()
                val_loss = 0.
                true_labels = []
                predicted_labels = []
                correct = 0
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
                

        print(f'{nan_ctr} NaNs')
        print('Finished Training')

    net = get_model(args, n_classes=n_classes).to(device)
    net.load_state_dict(torch.load(model_path, weights_only=False))
    print('loaded from', model_path)

    ## EVAL ##
    total_loss = 0.
    total_correct = 0
    n_examples = 0
    all_labels = []
    predicted_labels = []
    regr_labels = []
    regr_preds = []
    net.eval()

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
            if args.classification:
                if args.combined_loss:
                    tmp_labels = labels.flatten()
                    clf_labels = tmp_labels[1::2].long()
                    all_labels += clf_labels.tolist()
                    regr_labels += tmp_labels[::2].tolist()
                    _, predicted = torch.max(outputs[:, 1:], 1)
                    regr_preds += outputs[:, 0].flatten().tolist()
                    total_correct += (predicted == clf_labels).sum().item()
                else:
                    all_labels += labels.tolist()
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == labels).sum().item()
                predicted_labels += predicted.tolist()
            else:
                all_labels += labels.tolist()
                predicted_labels += outputs.tolist()

    if args.combined_loss:
        test_acc = total_correct / n_examples
        print(f'Accuracy: {test_acc}')
        r2 = r2_score(regr_labels, regr_preds)
        print(f'R2: {r2}')
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
        plt.scatter(all_labels, predicted_labels)
        plt.xlabel(f"true {args.dataset_label_type}")
        plt.ylabel(f"predicted {args.dataset_label_type}")
        plt.savefig(f"./imgs/{save_path}.png")
        plt.show()

    if args.gradcam:
        newtestloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                                    shuffle=shuffle, num_workers=2)        

        target_layers = [net.layer4[-1]]
        example = next(iter(newtestloader))
        # input_img, target = example[0][1].unsqueeze(0).to(device), example[1][1]
        input_img, target = example[0].to(device), example[1]
        print(input_img.size())

        target_class = args.gradcam_target_class

        if target_class == -1:
            targets = None
        else:
            targets = [ClassifierOutputReST(target_class)] * len(test_set)
        with GradCAM(model=net, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_img, targets=targets)
            print(grayscale_cam.shape)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam_avg = np.mean(grayscale_cam, axis=0)
            plt.imshow(grayscale_cam_avg)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("")
            plt.ylabel("")
            plt.title("GradCAM avg")
            targetclasstext = f"class_{target_class}" if target_class != -1 else "avg"
            plt.savefig(f"./imgs/gradcam_{targetclasstext}_{save_path}.png")

            # for i in range(3):
            #     grayscale_cam_img = grayscale_cam[i, :]
            #     # You can also get the model outputs without having to redo inference
            #     # model_outputs = cam.outputs
            #     plt.imshow(grayscale_cam_img)
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.xlabel("")
            #     plt.ylabel("")
            #     plt.title(f"GradCAM {i}")
            #     plt.savefig(f"./imgs/gradcam_{i}_{save_path}.png")
            

        newtestloader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                    shuffle=shuffle, num_workers=2)
        example = next(iter(newtestloader))
        input_img, target = gradcam_helper(example[0]).to(device), int(example[1])
        targets = [ClassifierOutputReST(target)] * input_img.size()[0]

        print(input_img[0][0])
        print(input_img[0][1])
        print(input_img[1][1])
        print(input_img[1][0])

        with GradCAM(model=net, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_img, targets=targets)
            gradcam_per_channel = np.mean(grayscale_cam, axis=(1,2))
            print(gradcam_per_channel)    
            plt.clf()
            plt.plot(range(len(gradcam_per_channel)), gradcam_per_channel)
            plt.savefig(f"./imgs/gradcam_channels_{save_path}.png")

def gradcam_helper(input_img):
    input_img = input_img.squeeze()
    n_channels = input_img.size()[0]
    input_img = torch.stack(tuple([input_img] * n_channels))
    print(input_img.size())
    for i in range(n_channels):
        for j in range(n_channels):
            if j == i: continue
            input_img[i][j] = 0.
        #TEST:
        # input_img[i][0] = input_img[i][i]
        # if i != 0: input_img[i][i] = 0.
    return input_img

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
    parser.add_argument("--gradcam", action='store_true')
    parser.add_argument("--gradcam_target_class", type=int, default=-1)
    parser.add_argument("--combined_loss", action='store_true') #regression with classification as regularizer

    args = parser.parse_args()
    print(args)

    train(args)


if __name__ == "__main__":
    main()
