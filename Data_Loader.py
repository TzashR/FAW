import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


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


# TODO update calc based on what Opher says.
def GRVI_calc(r, g):
    '''
    calculates GRVI index
    https://www.researchgate.net/publication/47426815_Applicability_of_Green-Red_Vegetation_Index_for_Remote_Sensing_of_Vegetation_Phenology

    '''
    if (g + r) == 0: return 0
    return (g - r) / (g + r)


def add_GRVI_channel(image_arr):
    '''

    @param image_arr: np.array of an image
    @return: the image with an extra channel computed by (green-red)/(green+red)
    '''
    calc_arr = np.zeros((image_arr.shape[0], image_arr.shape[1], 1))
    for i in range(image_arr.shape[0]):
        for j in range(image_arr.shape[1]):
            r = image_arr[i, j, 0]
            g = image_arr[i, j, 1]
            calc_arr[i, j, 0] = GRVI_calc(r, g)
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
    new_img = add_GRVI_channel(new_img)

    return new_img

class FawDataset(torch.utils.data.Dataset):
    def __init__(self, usable_reports, image_dim, transform=(lambda img: img)):
        self.images = usable_reports
        self.image_dim = image_dim
        if (transform == faw_transform):
            self.transform = lambda img: faw_transform(img, image_dim[0], image_dim[1])
        else:
            self.transform = transform

    def __getitem__(self, index):
        im_path = self.images[index]
        img = cv2.imread(im_path)
        return self.transform(img)

    def __len__(self):
        len(self.reports.items())



def make_batches(batch_size,reports_dic, min_images = 5, seed = 0):
    #list of all reports with at least min_images images
    usable_reports = list(filter(lambda report: len(report[1]['images']) >= min_images, reports_dic.items()))


    sizes_arr = [len(report[1]['images']) for report in usable_reports]
    batches_sizes = []
    used_set= set()
    used_sizes_sum = 0
    while used_sizes_sum<sum(sizes_arr)-max(sizes_arr):
        current_batch = []
        for i in range(len(sizes_arr)):
            if not (i in used_set):
                current_batch.append(sizes_arr[i])
                used_set.add(i)
                used_sizes_sum += sizes_arr[i]
                if sum(current_batch)>batch_size: break
        batches_sizes.append(current_batch)

    #creates a dictionary of {size: all reports with this amount of images}
    sizes_dic = {}
    for report in usable_reports:
        size = len(report[1]['images'])
        if size in sizes_dic:
            sizes_dic[size].append(report[0])
        else:
            sizes_dic[size] = [report[0]]

    #chooses random of ids of wantes sizes for batches
    id_batches = []
    random.seed(seed)
    for i in range(len(batches_sizes)):
        id_batches.append([])
        for size in batches_sizes[i]:
            reports_of_size = sizes_dic[size]
            random_index = random.randrange(len(reports_of_size))
            id_batches[i].append(reports_of_size.pop(random_index))

    #translates the batches ids to actual image indices
    batches = []
    for i in range(len(id_batches)):
        batches.append([])
        for report_id in id_batches[i]:
            image_indices = [img_loc[1] for img_loc in reports_dic[report_id]['images']]
            batches[i]+= image_indices
    return batches



def faw_batch_sampler(batch_size,reports_dic, min_images = 5, seed = 0):
    pass


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
ds = FawDataset(usable_reports_lst,(512,512,4),faw_transform)

x = make_batches(20,usable_reports_dic)
print(x)