DATASET_DIR = 'transfer_2997031_files_39a53adc/'

import pandas as pd
from PIL import Image
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy
from scipy import ndimage
from torch import save
from sklearn.metrics import r2_score


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
        if img_type=='none':
            self.img = None
        else:
            self.img = excel_line_to_img(excel_line, img_type=img_type)

    def __str__(self):
        return f'Kiwi {self.id}' #TODO add more info


def make_kiwi_dataset(excel_lines, sample_type='rgb', label_type='brix', n_samples=100, sample_idx=7):
    samples = []
    labels = []

    print(sample_idx * n_samples, (sample_idx + 1) * n_samples)

    for i in range(sample_idx * n_samples, (sample_idx + 1) * n_samples):
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


def excel_to_plots(excel_lines):
    n_samples = len(excel_lines)

    kiwis = []

    for i in range(n_samples):
        kiwi = Kiwi(excel_lines.iloc[i], img_type='none')
        kiwis.append(kiwi)

    brixes = {}
    penetros = {}
    awetas = {}
    
    brixes_l = []
    penetros_l = []
    awetas_l = []


    for kiwi in kiwis:        
        if kiwi.date not in brixes.keys():
            brixes[kiwi.date] = [kiwi.brix]
        else:
            brixes[kiwi.date].append(kiwi.brix)

        if kiwi.date not in penetros.keys():
            penetros[kiwi.date] = [kiwi.penetro]
        else:
            penetros[kiwi.date].append(kiwi.penetro)

        if kiwi.date not in awetas.keys():
            awetas[kiwi.date] = [kiwi.aweta]
        else:
            awetas[kiwi.date].append(kiwi.aweta)
        
        brixes_l.append(kiwi.brix)
        penetros_l.append(kiwi.penetro)
        awetas_l.append(kiwi.aweta)
    
    for date in penetros.keys():
        plt.scatter(penetros[date], awetas[date])
    
    plt.xlabel('penetro')
    plt.ylabel('aweta')
    plt.show()


    for date in penetros.keys():
        plt.scatter(penetros[date], brixes[date])
    
    plt.xlabel('penetro')
    plt.ylabel('brix')
    plt.show()

    print(awetas_l[:10])
    print(penetros_l[:10])
    print('r2 brix-aweta:', r2_score(brixes_l, awetas_l))
    print('r2 aweta-brix:', r2_score(awetas_l, brixes_l))
    print('r2 brix-penetro:', r2_score(brixes_l, penetros_l))
    print('r2 penetro-brix:', r2_score(penetros_l, brixes_l))
    print('r2 aweta-penetro:', r2_score(awetas_l, penetros_l))
    print('r2 penetro-aweta:', r2_score(penetros_l, awetas_l))

    awetas_l = np.array(awetas_l)
    awetas_l -= np.mean(awetas_l)
    awetas_l /= np.std(awetas_l)    
    penetros_l = np.array(penetros_l)
    penetros_l -= np.mean(penetros_l)
    penetros_l /= np.std(penetros_l)    
    brixes_l = np.array(brixes_l)
    brixes_l -= np.mean(brixes_l)
    brixes_l /= np.std(brixes_l)

    print(awetas_l[:10])
    print(penetros_l[:10])
    print('r2 brix-aweta:', r2_score(brixes_l, awetas_l))
    print('r2 aweta-brix:', r2_score(awetas_l, brixes_l))
    print('r2 brix-penetro:', r2_score(brixes_l, penetros_l))
    print('r2 penetro-brix:', r2_score(penetros_l, brixes_l))
    print('r2 aweta-penetro:', r2_score(awetas_l, penetros_l))
    print('r2 penetro-aweta:', r2_score(penetros_l, awetas_l))


def main():
    excel_lines_dataset = pd.read_excel(DATASET_DIR + 'HSI_dataset_info.xlsx')
    dataset = KiwiDataset(excel_lines_dataset, sample_type='spec')
    save(dataset, './data/kiwi_dataset_700-800.pt')
    print(len(dataset))

    # excel_to_plots(excel_lines_dataset)


if __name__ == "__main__":
    main()

