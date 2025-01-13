DATASET_DIR = 'transfer_2997031_files_39a53adc/'

import pandas as pd
from PIL import Image
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy

def excel_line_to_img(excel_line, img_type='rgb'):
    filepath = excel_line['cube_loc'].split('/')[-2:]

    path = DATASET_DIR + ('/').join(filepath) + '/results/'
    # print(path)

    img = Image.open(path + 'REFLECTANCE_' + filepath[1] + '.png')

    img = img.rotate(90)

    img_np = np.array(img)

    R = img_np[:,:,0]
    G = img_np[:,:,1]
    B = img_np[:,:,2]

    mask = (R < 15) & (G < 15) & (B < 15)

    img_np[mask] = [255,255,255,255]

    h,w,c = img_np.shape

    # print(excel_line['mask_label'])

    if (img_type == 'rgb'):
        full_img = img_np
    elif (img_type == 'spec'):
        full_img = envi.open(path + 'REFLECTANCE_' + filepath[1] + '.hdr', path + 'REFLECTANCE_' + filepath[1] + '.dat')

    if (excel_line['mask_label'] == 1):
        # quarter = full_img[:h//2, :w//2]
        # quarter_rgb = img_np[:h//2, :w//2]
        quarter = full_img[:h//2, w//2:] 
        quarter_rgb = img_np[:h//2, w//2:] 
    elif (excel_line['mask_label'] == 2):
        # quarter = full_img[:h//2, w//2:] 
        # quarter_rgb = img_np[:h//2, w//2:] 
        quarter = full_img[h//2:, w//2:]
        quarter_rgb = img_np[h//2:, w//2:]
    elif (excel_line['mask_label'] == 3):
        # quarter = full_img[h//2:, :w//2]
        # quarter_rgb = img_np[h//2:, :w//2]
        quarter = full_img[:h//2, :w//2]
        quarter_rgb = img_np[:h//2, :w//2]
    elif (excel_line['mask_label'] == 4):
        # quarter = full_img[h//2:, w//2:]
        # quarter_rgb = img_np[h//2:, w//2:]
        quarter = full_img[h//2:, :w//2]
        quarter_rgb = img_np[h//2:, :w//2]

    # plt.imshow(quarter[:,:,:4], cmap='gray')
    # plt.show()

    # plt.imshow(quarter_rgb[:,:,:4], cmap='gray')
    # plt.show()

    return quarter


class Kiwi:
    def __init__(self, excel_line, img_type='spec'):
        self.id = excel_line['code']
        self.date = excel_line['date']
        self.brix = excel_line['brix']
        self.penetro = excel_line['penetro']
        self.aweta = excel_line['aweta']
        self.treatment = excel_line['treatment']

        # print(excel_line)
        self.img = excel_line_to_img(excel_line, img_type=img_type)

    def __str__(self):
        return f'Kiwi {self.id}' #TODO add more info


def make_kiwi_dataset(excel_lines, sample_type='rgb', label_type='brix'):
    samples = []
    labels = []

    n_samples = 10 # TODO change to nr of excel lines

    for i in range(n_samples):
        kiwi = Kiwi(excel_lines.iloc[i], img_type=sample_type)
        samples.append(kiwi.img)
        if (label_type == 'brix'):
            labels.append(kiwi.brix)
        elif (label_type == 'penetro'):
            labels.append(kiwi.penetro)        
        elif (label_type == 'aweta'):
            labels.append(kiwi.aweta)
    
    samples = np.array(samples)
    labels = np.array(labels)

    return from_numpy(samples), from_numpy(labels)

class KiwiDataset(Dataset):
    def __init__(self, excel_lines, sample_type='rgb', label_type='brix'):
        super(KiwiDataset, self).__init__()
        self.samples, self.labels = make_kiwi_dataset(excel_lines, 
                                                      sample_type=sample_type, 
                                                      label_type=label_type)
        print (self.samples.size())
    
    def __getitem__(self, index):
        return self.samples[index], self.labels[index].float()

    def __len__(self):
        return self.samples.shape[0]


def main():
    excel_lines_dataset = pd.read_excel(DATASET_DIR + 'HSI_dataset_info.xlsx')
    dataset = KiwiDataset(excel_lines_dataset, sample_type='spec')
    print(len(dataset))

if __name__ == "__main__":
    main()
