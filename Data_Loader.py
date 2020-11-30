import os

import cv2
import numpy as np
import pandas as pd
import torch


def fix_id(id, i=-1):
    # check_id(i, id)
    return (id + 50) // 100 * 100


def check_id(i, id):
    # print(i, id)
    id1 = (id // 100) * 100
    # print(id1)
    assert id == id1 or (id - 4) == id1 or (id - 8) == id1 or (id - 96) == id1 or (id - 92) == id1


def make_actual_file_path(measurement_path, root_folder):
    '''
    @param measurement_path: A file path from photo conversion excel
    @param root_folder: Folder containing images
    @return: path to actual file in the folder
    '''

    splitted_measurement_path = measurement_path.split("/")
    actual_path = os.path.join(root_folder, splitted_measurement_path[2], splitted_measurement_path[3])

    return actual_path


# TODO I need to get the index for each image
def process_data(reports_df, images_df, images_dir):
    '''

    @param reports_table:
    @param images_table:
    @param images_dir:
    @return: -
        usable : { report_id: (infest_level, [image_file_path])}
        seedling_reports : {report id: [image_file_path}
        number of missing images (in reports but not in image dir)
        number of reportless images (in dir but not in reoprts)
        number of total images in dir


    '''
    reports_dic = {}
    seedlings_dic = {}

    # creates dictionary for reports (one for usable  and on for seedlings)
    for i in range(1, len(reports_df)):
        report_id = fix_id(fix_id(reports_df[0][i]))
        if reports_df[9][i] == 'Seedling':
            seedlings_dic[report_id] = []
        else:
            reports_dic[report_id] = {'infest_level': reports_df[7][i], 'images': []}

    num_images_not_in_reports = 0
    num_missing_image_files = 0
    total_images = 0
    usable_index = 0
    usable_reports_lst = []

    # assigns image files to their respective report in the dictionaries
    for i in range(1, len(images_df)):
        measurement_id = fix_id(images_df[0][i])
        image_file_path = make_actual_file_path(images_df[1][i], images_dir)

        if not os.path.isfile(image_file_path):
            num_missing_image_files += 1
            continue
        if measurement_id in reports_dic:
            usable_reports_lst.append(image_file_path)
            reports_dic[measurement_id]['images'].append((image_file_path, usable_index))
            usable_index += 1
        elif measurement_id in seedlings_dic:
            seedlings_dic[measurement_id].append(image_file_path)
        else:
            num_images_not_in_reports += 1
        total_images += 1

    return (reports_dic, usable_reports_lst, seedlings_dic,
            num_missing_image_files, num_images_not_in_reports, total_images)


# %%
def generate_batch(batch_size, seed=0):
    '''
    @param batch_size:
    @param seed:
    @return:
    '''


def faw_transform(image, wanted_dims):
    image_array = np.array(image)


class FawDataset(torch.utils.data.Dataset):
    def __init__(self, usable_reports, image_dim, transform=None):
        self.images = usable_reports
        self.image_dim = image_dim
        if (transform == faw_transform):
            self.transform = lambda image_arr: faw_transform(image_arr, image_dim[0], image_dim[1])
        elif transform is not None:
            self.transform = transform

    def __getitem__(self, index):
        im_path = self.images[index]
        image = cv2.imread(im_path)

        pass

    def __len__(self):
        len(self.reports.items())


# TODO add Dataloader
# TODO calculate 4th channel

# %%
'''
Data processing cell
'''
reports_file = "D:\Kenya IPM field reports.xls"
images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
reports_df = pd.read_excel(reports_file, header=None)
id_infest_list = []
n_rows = len(reports_df[0][:])

id_infest_list = [(fix_id(i + 2, t[0]), t[1]) for i, t in enumerate(id_infest_list)]
id_to_infest = dict(id_infest_list)
images_df = pd.read_excel(images_file, header=None)

USB_PATH = r"D:\2019_clean2"

usable_reports_dic, usable_reports_lst, seedling_reports, missing_image_files, reportless_images, total_images = process_data(
    reports_df, images_df, USB_PATH)

# %%
image = cv2.imread(usable_reports_lst[0])
image_arr = np.array(image)
zeros_arr = np.zeros((image_arr.shape[0], image_arr.shape[1], 1))
for i in range(image_arr.shape[0]):
    for j in range(image_arr.shape[1]):
        r = image_arr[i, j, 0]
        g = image_arr[i, j, 1]
        zeros_arr[i, j, 0] = (int(r) + int(g)) / 2


# TODO update calc based on what Opher says.
def index_calc(r, g):
    '''

    @param r: red value from pixel
    @param g: green value from pixel
    @return: index calc
    '''
    return 1


def add_rg_channel(image_arr):
    '''

    @param image_arr: np.array of an image
    @return: the image with an extra channel computed by:...
    '''
    calc_arr = np.zeros((image_arr.shape[0], image_arr.shape[1], 1))
    for i in range(image_arr.shape[0]):
        for j in range(image_arr.shape[1]):
            r = int(image_arr[i, j, 0])
            g = int(image_arr[i, j, 1])
            calc_arr[i, j, 0] = index_calc(r, g)
    return np.concatenate((image_arr, calc_arr), axis=2)


def resize_image(image_arr, dim1, dim2):
    img = image_arr / 255.0
    if img.shape[0] < img.shape[1]:  # height first
        img = np.transpose(img, (1, 0, 2))
    img = cv2.resize(img, (dim1, dim2))
    return img


def faw_transform(img, dim1, dim2):
    new_img = resize_image(img, dim1, dim2)
    new_img = np.array(new_img)
    new_img = add_rg_channel(new_img)

    return new_img
