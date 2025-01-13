DATASET_DIR = 'transfer_2997031_files_39a53adc/'

import pandas as pd
from PIL import Image
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt

test = pd.read_excel(DATASET_DIR + 'HSI_dataset_info.xlsx')

def excel_line_to_img(excel_line):
    filepath = excel_line['cube_loc'].split('/')[-2:]

    path = DATASET_DIR + ('/').join(filepath) + '/results/'
    print(path)

    img = Image.open(path + 'REFLECTANCE_' + filepath[1] + '.png')

    img = img.rotate(180)

    img_np = np.array(img)

    R = img_np[:,:,0]
    G = img_np[:,:,1]
    B = img_np[:,:,2]

    mask = (R < 15) & (G < 15) & (B < 15)

    img_np[mask] = [255,255,255,255]

    h,w,c = img_np.shape

    print(excel_line['mask_label'])

    if (excel_line['mask_label'] == 1):
        quarter = img_np[:h//2, :w//2]
    elif (excel_line['mask_label'] == 2):
        quarter = img_np[:h//2, w//2:]    
    elif (excel_line['mask_label'] == 3):
        quarter = img_np[:h//2, :w//2]
    elif (excel_line['mask_label'] == 4):
        quarter = img_np[h//2:, w//2:]

    plt.imshow(quarter, cmap='gray')
    plt.show()

    return quarter

    # print(img_np)

    # print(img_np.shape)
    # 
    # spec = envi.open(path + 'REFLECTANCE_' + filepath[1] + '.hdr', path + 'REFLECTANCE_' + filepath[1] + '.dat')

    #TODO knip in kwarten corresponding aan idx..


class Kiwi:
    def __init__(self, excel_line):
        self.id = excel_line['code']
        self.date = excel_line['date']
        self.brix = excel_line['brix']
        self.penetro = excel_line['penetro']
        self.aweta = excel_line['aweta']
        self.treatment = excel_line['treatment']

        print(excel_line)
        self.img = excel_line_to_img(excel_line)

    def __str__(self):
        return f'Kiwi {self.id}' #TODO add more info




a = Kiwi(test.iloc[13])
print(a)