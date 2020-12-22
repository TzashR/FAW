import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
    index_to_label = {}

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
            index_to_label[usable_index] = reports_dic[measurement_id]['infest_level']
            usable_index += 1
        elif measurement_id in seedlings_dic:
            seedlings_dic[measurement_id].append(image_file_path)
        else:
            num_images_not_in_reports += 1
        total_images += 1

    return (reports_dic, usable_reports_lst, index_to_label, seedlings_dic,
            num_missing_image_files, num_images_not_in_reports, total_images)


def faw_transform(img):
    new_img = np.zeros((img.shape[0], img.shape[1], 4))
    new_img[:, :, :3] = img / 255.0
    # adds GRVI channel. (r+g)/(r-g)
    new_img[:, :, 3] = (new_img[:, :, 1] - new_img[:, :, 0]) / (new_img[:, :, 0] + new_img[:, :, 1] + .00001)
    if new_img.shape[0] < new_img.shape[1]:  # height first
        new_img = np.transpose(new_img, (1, 0, 2))
    new_img = np.transpose(new_img, (2, 0, 1))
    assert (new_img.shape == (4,1600,960))
    return new_img


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
        return img, label

    def __len__(self):
        len(self.images.items())

def make_batches(batch_size, reports_dic, min_images=5, seed=0):
    # list of all reports with at least min_images images
    usable_reports = list(filter(lambda report: len(report[1]['images']) >= min_images, reports_dic.items()))
    random.shuffle(usable_reports)
    n = len(usable_reports)
    num_of_batches,remainder = n//batch_size, n%batch_size

    #batches at this point are reports, we need to convert them to image indices
    batch_reports = [usable_reports[i:i+batch_size] for i in range(num_of_batches)]
    if remainder>0: batch_reports.append(usable_reports[-remainder:])

    batches = []
#converts all reports in batches to a list of image indices
    for rep_batch in batch_reports:
        batch_indices = []
        for report in rep_batch:
            for image_tuple in report[1]['images']: #image tuple contains path, index
                batch_indices.append(image_tuple[1])
        batches.append(batch_indices)
    return batches


def faw_batch_sampler(batches):
    for i in range(len(batches)):
        yield batches[i]


# # %%
# ##for testing
#
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
# batches2 = make_batches2(5, usable_reports_dic)
# sampler = faw_batch_sampler(batches)
#
# dl = DataLoader(ds, batch_sampler=sampler)
