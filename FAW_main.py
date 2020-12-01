import pandas as pd
from torch.utils.data import DataLoader

from Data_Loader import process_data, faw_batch_sampler, make_batches, FawDataset, faw_transform


def main():
    # user defined variables:
    reports_file = "D:\Kenya IPM field reports.xls"
    images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
    images_root_directory = r"D:\2019_clean2"
    training_set_size = 0.8  # how many (out of 1) should be in the training set.
    batch_size = 20

    val_set_size = 1 - training_set_size
    assert (val_set_size > 0 and val_set_size + training_set_size == 1)

    ##get and process the data
    reports_df = pd.read_excel(reports_file, header=None)
    images_df = pd.read_excel(images_file, header=None)


    usable_reports_dic, usable_reports_lst, index_to_label, seedling_reports, missing_image_files, reportless_images, total_images = process_data(
        reports_df, images_df, images_root_directory)

    ds = FawDataset(images=usable_reports_lst, labels=index_to_label, transform=faw_transform)

    all_batches = make_batches(batch_size, usable_reports_dic)

    train_until_index = int(len(all_batches)*training_set_size)

    train_sampler = faw_batch_sampler(all_batches[:train_until_index])
    val_sampler = faw_batch_sampler(all_batches[train_until_index:])

    train_dl = DataLoader(ds, batch_sampler=train_sampler)
    val_dl = DataLoader(ds, batch_sampler=val_sampler)



if __name__ == '__main__':
    main()
