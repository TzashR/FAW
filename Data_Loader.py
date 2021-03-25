import pickle

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from services import process_data, make_batches


class FawDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=(lambda img: img)):
        if (transform == faw_transform):
            self.transform = lambda img: faw_transform(img)
        else:
            self.transform = transform

        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        im_path = self.images[index]
        img = cv2.imread(im_path)
        label = self.labels[index]
        img = self.transform(np.array(img))
        img = torch.FloatTensor(img)

        return img, label

    def __len__(self):
        len(self.images.items())

def hello():
    return "hello"

def faw_transform(img):
    new_img = np.zeros((img.shape[0], img.shape[1], 4))
    new_img[:, :, :3] = img / 255.0
    # adds GRVI channel. (r+g)/(r-g)
    new_img[:, :, 3] = (new_img[:, :, 1] - new_img[:, :, 0]) / (new_img[:, :, 0] + new_img[:, :, 1] + .00001)
    if new_img.shape[0] < new_img.shape[1]:  # height first
        new_img = np.transpose(new_img, (1, 0, 2))
    new_img = np.transpose(new_img, (2, 0, 1))
    assert (new_img.shape == (4, 1600, 960)), f'new_img shape {new_img.shape}, img shape {img.shape} img = {img}'
    return new_img


def make_ds_and_batches(reports_file, images_file, images_root_directory, mode, bad_shape_images_path=None,
                        batch_size=None):
    '''
    Creates the data set that can be used in train or test. if is_train  then the function also returns a list of all the batches indices
    @param reports_file: File path to the reports table
    @param images_file:  File path to the images table
    @param images_root_directory: Directory where images a stored
    @param mode: Is this a train or test session. 'test' for test 'train' for train.
    @param bad_shape_images_path:
    @param batch_size:
    @return:
    '''
    assert (mode == 'train' or mode == 'test')
    assert (not (mode == 'train' and batch_size is None))

    if bad_shape_images_path is not None:
        with open(bad_shape_images_path, 'rb') as fp:
            bad_shape_images = pickle.load(fp)
    else:
        bad_shape_images = None

    #get and process the data
    reports_df = pd.read_excel(reports_file, header=None)
    images_df = pd.read_excel(images_file, header=None)
    usable_reports_dic, usable_reports_lst, index_to_label, seedling_reports, missing_image_files, reportless_images, total_images \
        = process_data(
        reports_df, images_df, images_root_directory, bad_shape_images)
    ds = FawDataset(images=usable_reports_lst, labels=index_to_label, transform=faw_transform)
    if mode == 'train':
        all_batches = make_batches(batch_size, usable_reports_dic)
        return ds, all_batches
    else:
        return ds

# %%
##for testing

# reports_file = "D:\Kenya IPM field reports.xls"
# images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
# reports_df = pd.read_excel(reports_file, header=None)
# images_df = pd.read_excel(images_file, header=None)
#
# USB_PATH = r"D:\2019_clean2"
#
# usable_reports_dic, usable_reports_lst, index_to_label, seedling_reports, missing_image_files, reportless_images, total_images = process_data(
#     reports_df, images_df, USB_PATH)
#
# ds = FawDataset(images=usable_reports_lst, labels=index_to_label, transform=faw_transform)
#
# batches = make_batches(20, usable_reports_dic)
# batches2 = make_batches(5, usable_reports_dic)
# sampler = faw_batch_sampler(batches)
#
# dl = DataLoader(ds, batch_sampler=sampler)
