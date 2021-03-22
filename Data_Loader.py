import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

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

def faw_transform(img):
    new_img = np.zeros((img.shape[0], img.shape[1], 4))
    new_img[:, :, :3] = img / 255.0
    # adds GRVI channel. (r+g)/(r-g)
    new_img[:, :, 3] = (new_img[:, :, 1] - new_img[:, :, 0]) / (new_img[:, :, 0] + new_img[:, :, 1] + .00001)
    if new_img.shape[0] < new_img.shape[1]:  # height first
        new_img = np.transpose(new_img, (1, 0, 2))
    new_img = np.transpose(new_img, (2, 0, 1))
    assert (new_img.shape == (4,1600,960)), f'new_img shape {new_img.shape}, img shape {img.shape} img = {img}'
    return new_img

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
