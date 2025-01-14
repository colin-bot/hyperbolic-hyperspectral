DATASET_DIR = 'transfer_2997031_files_39a53adc/'

import pandas as pd
from PIL import Image
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
from scipy import ndimage


def excel_line_to_img(excel_line, img_type='rgb'):
    filepath = excel_line['cube_loc'].split('/')[-2:]
    path = DATASET_DIR + ('/').join(filepath) + '/results/'
    # print(path)

    # read rgb image and rotate to align with hyperspec
    img = Image.open(path + 'REFLECTANCE_' + filepath[1] + '.png')
    img = img.rotate(90) # to align orientation with the spectral image
    img_np = np.array(img)

    # images (rgb & hyperspectral) to be sliced:
    full_img_rgb = img_np
    full_img = envi.open(path + 'REFLECTANCE_' + filepath[1] + '.hdr', path + 'REFLECTANCE_' + filepath[1] + '.dat')

    R = img_np[:,:,0]
    G = img_np[:,:,1]
    B = img_np[:,:,2]

    mask = (R < 15) & (G < 15) & (B < 15)

    img_np[mask] = [255,255,255,255] # set all dark pixels to be very light (like the label papers)

    # crop the correct kiwi. the image is rotated so it isnt just 1 = topleft, 2=topright etc but instead:
    # topleft = 3, topright = 1, bottomleft=4, bottomright=2. commented slices are right-side-up
    quarter_margin = 20
    h,w,c = img_np.shape
    # print(excel_line['mask_label'])
    if (excel_line['mask_label'] == 1):
        # quarter = full_img[:h//2, :w//2]
        # quarter_rgb = img_np[:h//2, :w//2]
        quarter = full_img[:h//2+quarter_margin, w//2-quarter_margin:] 
        quarter_rgb = full_img_rgb[:h//2+quarter_margin, w//2-quarter_margin:] 
    elif (excel_line['mask_label'] == 2):
        # quarter = full_img[:h//2, w//2:] 
        # quarter_rgb = img_np[:h//2, w//2:] 
        quarter = full_img[h//2-quarter_margin:, w//2-quarter_margin:]
        quarter_rgb = full_img_rgb[h//2-quarter_margin:, w//2-quarter_margin:]
    elif (excel_line['mask_label'] == 3):
        # quarter = full_img[h//2:, :w//2]
        # quarter_rgb = img_np[h//2:, :w//2]
        quarter = full_img[:h//2+quarter_margin, :w//2+quarter_margin]
        quarter_rgb = full_img_rgb[:h//2+quarter_margin, :w//2+quarter_margin]
    elif (excel_line['mask_label'] == 4):
        # quarter = full_img[h//2:, w//2:]
        # quarter_rgb = img_np[h//2:, w//2:]
        quarter = full_img[h//2-quarter_margin:, :w//2+quarter_margin]
        quarter_rgb = full_img_rgb[h//2-quarter_margin:, :w//2+quarter_margin]

    # invert the image so that the kiwi becomes very bright
    test = np.mean(quarter_rgb, axis=2)
    test = np.full(shape=test.shape, fill_value=255) - test

    # isolate the label paper edges & writing
    test[test < 100] = 0

    # gauss blur to make thin bright lines become soft dark lines
    test = ndimage.gaussian_filter(test, sigma=5)

    # to remove the softened lines
    test[test < 100] = 0

    # print(test.shape)
    # plt.imshow(test)
    # plt.show()

    # find center of mass
    half_size=90
    c_o_m = ndimage.center_of_mass(test)

    com0_clamped = max(half_size, min(c_o_m[0], test.shape[0]-half_size))
    com1_clamped = max(half_size, min(c_o_m[1], test.shape[1]-half_size))

    if (img_type=='rgb'):
        final_crop = quarter_rgb[int(com0_clamped-half_size):int(com0_clamped+half_size), 
                                 int(com1_clamped-half_size):int(com1_clamped+half_size),
                                 :]
    
        plt.imshow(final_crop)
        plt.show()
    elif (img_type=='spec'):
        final_crop = quarter[int(com0_clamped-half_size):int(com0_clamped+half_size), 
                             int(com1_clamped-half_size):int(com1_clamped+half_size),
                             :]

    # print(final_crop.shape)

    return final_crop


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

    n_samples = 20 # TODO change to nr of excel lines

    for i in range(16,20):
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

