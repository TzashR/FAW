import os
from collections import defaultdict
import torch

import pandas as pd


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


def process_data(reports_df, images_df, images_dir):
    '''

    @param reports_table:
    @param images_table:
    @param images_dir:
    @return: - number of useable images, number of seedling images, number of images in the report but not in the folder
    '''
    reports_dic = {}
    seedlings_dic = {}

    for i in range(1, len(reports_df)):
        report_id = fix_id(fix_id(reports_df[0][i]))
        if reports_df[9][i] == 'Seedling':
            seedlings_dic[report_id] = []
        else:
            reports_dic[report_id] = {'infest_level': reports_df[7][i], 'images': []}

    num_images_not_in_reports = 0
    num_missing_image_files = 0
    total_images = 0

    for i in range(1, len(images_df)):
        measurement_id = fix_id(images_df[0][i])
        image_file_path = make_actual_file_path(images_df[1][i], images_dir)

        if not os.path.isfile(image_file_path):
            num_missing_image_files += 1
            continue
        if measurement_id in reports_dic:
            reports_dic[measurement_id]['images'].append(image_file_path)
        elif measurement_id in seedlings_dic:
            seedlings_dic[measurement_id].append(image_file_path)
        else:
            num_images_not_in_reports += 1
        total_images += 1

    return (seedlings_dic, reports_dic,
            num_missing_image_files, num_images_not_in_reports, total_images)


# %%
def generate_batch(batch_size, seed=0):
    '''
    @param batch_size:
    @param seed:
    @return:
    '''


class FawDataset(torch.utils.data.Dataset):
    def __init__(self,usealbe_reports, transform = None ):
        self.reports = useable_reports

    def __getitem__(self, item):
        # TODO ge item should somehow return all the images of a specific report
        pass
    def __len__(self):
        len(self.reports.items())

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

useable_reports, seedling_reports, missing_image_files, reportless_images, total_images = process_data(
    reports_df, images_df,USB_PATH)


