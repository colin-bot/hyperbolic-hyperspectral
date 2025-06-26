# Helper functions used to load alternate dataset (from deephs_fruit)

from core.datasets.hyperspectral_dataset import HyperspectralDataset, get_records
from core.name_convention import CameraType, ClassificationType, Fruit
from classification.transformers.normalize import Normalize
from classification.transformers.data_augmentation import Augmenter
from torch.utils.data import DataLoader


def load_deephs():
    AUGMENTATION_CONFIG_TRAIN = {
        'random_flip': True,
        'random_rotate': True,
        'random_noise': False,
        'random_cut': True,
        'random_crop': True,
        'random_intensity_scale': False
    }

    hparams = {'fruit': Fruit.AVOCADO,
               'camera_type': CameraType.VIS,
               'classification_type': ClassificationType.RIPENESS,
               'input_size': (64, 64),
               'data_path': '/scratch-shared/cbot/',
               'batch_size' : 4,
               'num_workers' : 4,
               'augmentation_config' : AUGMENTATION_CONFIG_TRAIN}

    train_records, val_records, test_records = \
        get_records(hparams['fruit'],
                hparams['camera_type'],
                hparams['classification_type'],
                use_inter_ripeness_levels=True,
                extend_by_time_assumption=True,
                allow_all_fruit_types=True)

    common_preprocessing = Normalize(hparams['camera_type'])

    train_preprocessing = []
    train_preprocessing += [Augmenter(augmentation_config=hparams['augmentation_config'],
                                      input_size=hparams['input_size'])]
    train_dataset = HyperspectralDataset(hparams['classification_type'], train_records,
                                              data_path=hparams['data_path'],
                                              balance=False,
                                              transform=common_preprocessing,
                                              input_size=hparams['input_size'])
    val_dataset = HyperspectralDataset(hparams['classification_type'],
                                            val_records,
                                            data_path=hparams['data_path'],
                                            transform=common_preprocessing,
                                            input_size=hparams['input_size'])
    test_dataset = HyperspectralDataset(hparams['classification_type'], test_records,
                                             data_path=hparams['data_path'],
                                             transform=common_preprocessing,
                                             input_size=hparams['input_size'])

    train_dataloader = DataLoader(train_dataset, hparams['batch_size'], num_workers=hparams['num_workers'], shuffle=True, drop_last=True, collate_fn=None)

    val_dataloader = DataLoader(val_dataset, 1, num_workers=hparams['num_workers'], shuffle=False)

    test_dataloader = DataLoader(test_dataset, 1, num_workers=hparams['num_workers'])

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, val_dataloader, test_dataloader = load_deephs()
    for i, data in enumerate(train_dataloader, 0):
        print(data[0].shape)
        print(data[1].shape)
        break
