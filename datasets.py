# https://www.youtube.com/watch?v=DuSqffoDojM -> 1:29:03
# -> https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4

from typing import Any, Callable, Optional

import torch
from torchvision.datasets import MNIST, SVHN, FashionMNIST

import numpy as np
import h5py
import os
import cv2

class Fast_MNIST(MNIST):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        # deleting device key from kwargs (as cannot be passed to MNIST mother class)
        device = kwargs['device']
        del kwargs['device']
        super().__init__(*args, **kwargs)

        # Insert channel dim and scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # binarize MNIST by sampling from Bernoulli with probs that are pixel intensities
        # done as in Salakhutdinov & Murray (2008)
        # https://dl.acm.org/doi/pdf/10.1145/1390156.1390266?casa_token=mvJb8zb-U1EAAAAA:fH7RMyQbJi5vGtrUySEmv_Raf19T4PxWRlyoWXoKi4uMBeJJtxLMjryTMJEe-mTgIRciqBVlBOA
        # n_examples, in_dim = self.data.size()[0], (self.data.size()[2] * self.data.size()[3])
        # self.data = torch.flatten(self.data)
        # bernoulli = torch.distributions.Bernoulli(self.data)
        # self.data = bernoulli.sample()
        # self.data = self.data.view(n_examples, in_dim)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Fast_SVHN(SVHN):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4
    """
    def __init__(self, *args, **kwargs):
        # deleting device key from kwargs (as cannot be passed to SVHN mother class)
        device = kwargs['device']
        del kwargs['device']
        super().__init__(*args, **kwargs)

        # convert numpy to torch
        self.data = torch.tensor(self.data)
        self.labels = torch.tensor(self.labels)

        # Scale data to [-.5, .5]
        # self.data = self.data.float().div(255) - .5

        # Scale data to [0., 1.]
        self.data = self.data.float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        # attribute is self.labels, see https://pytorch.org/docs/stable/_modules/torchvision/datasets/svhn.html#SVHN
        self.data, self.labels = self.data.to(device), self.labels.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index], self.labels[index]

        return img, label



class Fast_3DShapes(torch.utils.data.Dataset):
    """
    Source as modified from: https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4

    # Dataset from: https://github.com/deepmind/3d-shapes
    # Using loading and sampling script from: https://github.com/deepmind/3d-shapes/blob/master/3dshapes_loading_example.ipynb
    """
    def __init__(self, train, device,
                 train_frac=1.,
                 factors_variation_dict={'floor_hue': list(range(10)), 'wall_hue': list(range(10)), 'object_hue': list(range(10)), 'scale': list(range(8)), 'shape': list(range(4)), 'orientation': list(range(15))},
                 factors_label_list=['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'],
                 seed=0,
                 data_folder=None):
        # make the train-test split deterministic
        torch.manual_seed(seed)
        np.random.seed(seed)

        # deleting device key from kwargs (as cannot be passed to SVHN mother class)
        self.DATA_RAW_DIR = 'data/Fast_3DShapes/raw' if data_folder is None else os.path.join(data_folder, 'Fast_3DShapes')
        self.DATA_PROC_DIR = 'data/Fast_3DShapes/processed'
        self.RAW_FILE_NAME = '3dshapes.h5'

        # check if dataset is available
        if not os.path.exists(os.path.join(self.DATA_RAW_DIR, self.RAW_FILE_NAME)):
            exit("3DShapes not downloaded yet! Download the file 3dshapes.h5 from https://console.cloud.google.com/storage/browser/3d-shapes;tab=objects?prefix=&forceOnObjectsSortingFiltering=false (see https://github.com/deepmind/3d-shapes for further instructions) and put this file into data/Fast_3DShapes/raw")

        raw_file_path = os.path.join(self.DATA_RAW_DIR, self.RAW_FILE_NAME)

        # from loading script
        dataset = h5py.File(raw_file_path, 'r')
        # print(dataset.keys())
        data = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        labels = dataset['labels']  # array shape [480000,6], float64

        image_shape = data.shape[1:]  # [64,64,3]
        label_shape = labels.shape[1:]  # [6]
        n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000
        _FACTOR_TO_COLUMN_INDEX = {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}
        _FACTOR_TO_ALLOWED_VALUES = {'floor_hue': list(range(10)), 'wall_hue': list(range(10)), 'object_hue': list(range(10)), 'scale': list(range(8)), 'shape': list(range(4)), 'orientation': list(range(15))}

        _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                  'scale': 8, 'shape': 4, 'orientation': 15}

        # convert labels to numpy array
        labels = labels[()]
        labels = labels.astype(float)

        # convert float to int scales
        labels[:, 0] = labels[:, 0] * 10  # [0, 1, ..., 9]
        labels[:, 1] = labels[:, 1] * 10  # [0, 1, ..., 9]
        labels[:, 2] = labels[:, 2] * 10  # [0, 1, ..., 9]

        # label in column index 3 is very weird -> manual remapping required
        labels[:, 3] = np.round(labels[:, 3], 2)  #  first round, since values are very precise floats
        remap = {0.75: 0.0, 0.82: 1.0, 0.89: 2.0, 0.96: 3.0, 1.04: 4.0, 1.11: 5.0, 1.18: 6.0, 1.25: 7.0}
        label_3 = np.copy(labels[:, 3])
        for k, v in remap.items():
            label_3[labels[:, 3] == k] = v
        labels[:, 3] = label_3

        # shape is already on int scale   # # [0, 1, ..., 3]

        labels[:, 5] = np.round(labels[:, 5], 2)  #  first round, since values are very precise floats
        remap = {-30.: 0, -25.71: 1, -21.43: 2, -17.14: 3, -12.86: 4,  -8.57: 5,  -4.29: 6,   0.: 7, 4.29: 8,   8.57: 9,  12.86: 10,  17.14: 11,  21.43: 12,  25.71: 13,  30.: 14}
        label_5 = np.copy(labels[:, 5])
        for k, v in remap.items():
            label_5[labels[:, 5] == k] = v
        labels[:, 5] = label_5  # [0, 1, ..., 15]

        # make labels an int, because
        # since 3 in labels[:, 0] is actually 3.0000000000004, even though not correctly displayed
        labels = labels.astype(int)

        # some asserts
        assert _FACTOR_TO_COLUMN_INDEX.keys() == factors_variation_dict.keys()
        assert all(x in _FACTOR_TO_COLUMN_INDEX.keys() for x in factors_label_list)
        # assert that values of variation for each factor are correctly chosen
        for (key, value) in factors_variation_dict.items():
            assert all(x in _FACTOR_TO_ALLOWED_VALUES[key] for x in value)
        # each label factor chosen must have at least 2 values in the corresponding factors_variation_dict
        for factor in factors_label_list:
            assert len(factors_variation_dict[factor]) > 1
        # at least one label chosen
        assert len(factors_label_list) > 0
        # assert 'scale' not in factors_label_list  # because there is an issue with it

        # choose the data
        chosen_conjunction = None
        for i, (key, value) in enumerate(factors_variation_dict.items()):
            chosen = np.in1d(labels[:, _FACTOR_TO_COLUMN_INDEX[key]], value)
            if i == 0:
                chosen_conjunction = chosen
            else:
                chosen_conjunction = np.logical_and(chosen_conjunction, chosen)
        # chosen_conjunction = chosen_conjunction.flatten()
        data = data[chosen_conjunction, :, :, :]
        labels = labels[chosen_conjunction, :]

        # choose the label
        label_column_indices = []
        for factor in factors_label_list:
            label_column_indices.append(_FACTOR_TO_COLUMN_INDEX[factor])
        labels = labels[:, label_column_indices]

        # scale down the images from (64x64) to (32x32)
        scaled_images = []
        for i in range(data.shape[0]):
            img = data[i]
            img = cv2.resize(img, dsize=(32, 32))  # see https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
            scaled_images.append(img)
        data = np.asarray(scaled_images)

        # Scale data to [0., 1.]
        data = data / 255

        # split the data into train and test
        # this is currently done in a very simple way: just take the first train_frac*100 percent for train, rest for test
        # but it could be more complex: e.g. selecting only among the "pure variation"/non-label columns of labels to ensure that every label value is in both train and test
        # assuming that the factors_label_list shall contain rather few items compared to factors_variation_dict.keys(), this should not make a large difference
        # take a random sample
        train_indices = np.random.choice(data.shape[0], int(data.shape[0] * train_frac), replace=False)
        test_indices = np.setdiff1d(np.arange(data.shape[0]), train_indices)
        assert np.array_equal(np.setdiff1d(np.concatenate((train_indices, test_indices)), np.arange(data.shape[0])), np.array([]).astype(int))
        if train:
            data = data[train_indices]
            labels = labels[train_indices]
        else:
            data = data[test_indices]
            labels = labels[test_indices]

        # convert data to numpy (after slicing, sizing etc. is all done to be minimum size)
        print("converting to numpy. this can take more than one minute, depending on how much variation is selected...")
        data = data[()]
        # data = data.astype(float)

        # channel first -> transpose dimensions
        data = np.transpose(data, (0, 3, 1, 2))

        # convert numpy to torch
        data = torch.tensor(data)
        labels = torch.tensor(labels)

        # cast data to FLoatTensor
        data = data.float()

        # Put both data and targets on GPU in advance
        # attribute is self.labels, see https://pytorch.org/docs/stable/_modules/torchvision/datasets/svhn.html#SVHN
        data, labels = data.to(device), labels.to(device)

        # assign as instance variables
        self.data = data
        self.labels = labels


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.data[index], self.labels[index]

        return img, label


    def __len__(self):
        """
        """
        return self.data.shape[0]



if __name__ == "__main__":
    dataset = Fast_3DShapes(train=True, device='cuda:1', train_frac = 1.,
                                                         factors_variation_dict={'floor_hue': list(range(4)), 'wall_hue': list(range(4)), 'object_hue': list(range(4)), 'scale': list(range(4)), 'shape': list(range(4)), 'orientation': list(range(5))}, \
                                                         factors_label_list=['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'])

    # dataset = Fast_3DShapes(train=True, device='cuda:1', train_frac = 1.,
    #                                                  factors_variation_dict={'floor_hue': list(range(10)), 'wall_hue': list(range(10)), 'object_hue': list(range(10)), 'scale': list(range(8)), 'shape': list(range(4)), 'orientation': list(range(15))},
    #                                                  factors_label_list=['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'])


    _FACTOR_TO_COLUMN_INDEX = {'floor_hue': 0, 'wall_hue': 1, 'object_hue': 2, 'scale': 3, 'shape': 4, 'orientation': 5}