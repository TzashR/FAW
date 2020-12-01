import pandas as pd
from torch.utils.data import DataLoader

from Data_Loader import process_data, faw_batch_sampler, make_batches, FawDataset, faw_transform


def main():
    ##get and process the data
    reports_file = "D:\Kenya IPM field reports.xls"
    images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
    reports_df = pd.read_excel(reports_file, header=None)
    images_df = pd.read_excel(images_file, header=None)

    USB_PATH = r"D:\2019_clean2"

    usable_reports_dic, usable_reports_lst, index_to_label, seedling_reports, missing_image_files, reportless_images, total_images = process_data(
        reports_df, images_df, USB_PATH)

    ds = FawDataset(usable_reports_lst, index_to_label, (512, 512, 4), faw_transform)

    batches = make_batches(20, usable_reports_dic)
    sampler = faw_batch_sampler(batches)

    dl = DataLoader(ds, batch_sampler=sampler)


if __name__ == '__main__':
    main()
