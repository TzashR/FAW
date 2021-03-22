import argparse
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from CNN import FawNet
from Data_Loader import  FawDataset, faw_transform
import pickle
from services import process_data, faw_batch_sampler, make_batches, train_epochs
import os

def main():
    ##### receiving user input
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_file", help="File path to the reports table")
    parser.add_argument("images_table_file", help="File path to the image table")
    parser.add_argument("images_root_directory", help="path to images root directory")
    parser.add_argument("outputs_directory", help="path directory where outputs will be saved")
    parser.add_argument("--is_gpu", help="path to images root directory", type=bool, default=False)
    parser.add_argument("-pb","--with_pbar", help="Should have progress bar", type=bool, default=False)
    parser.add_argument("-pl","--print_losss", help="Should print loss while training", default=False)
    parser.add_argument("-bs","--bad_shapes", help="File with paths to bad images", default=None)
    args = parser.parse_args()

    reports_file = args.reports_file
    images_file = args.images_table_file
    images_root_directory = args.images_root_directory
    outputs_dir = args.outputs_directory
    is_gpu = args.is_gpu
    with_pbar = args.with_pbar
    print_loss = args.print_loss
    bad_shape_images_path = args.bad_shapes

    if bad_shape_images_path is not None:
        with open(bad_shape_images_path, 'rb') as fp:
            bad_shape_images = pickle.load(fp)
    else: bad_shape_images = None
    #######

    train_set_size = 0.75  # how many (out of 1) should be in the training set.
    test_set_size = 0.25
    val_set_size = 1 - test_set_size - train_set_size
    batch_size = 1
    epochs = 2

    assert (1 > test_set_size > 0 and 1 > train_set_size > 0 and train_set_size + test_set_size + val_set_size == 1)

    ##get and process the data
    reports_df = pd.read_excel(reports_file, header=None)
    images_df = pd.read_excel(images_file, header=None)

    usable_reports_dic, usable_reports_lst, index_to_label, seedling_reports, missing_image_files, reportless_images, total_images \
        = process_data(
        reports_df, images_df, images_root_directory, bad_shape_images)

    ds = FawDataset(images=usable_reports_lst, labels=index_to_label, transform=faw_transform)
    all_batches = make_batches(batch_size, usable_reports_dic)
    train_until_index = int(len(all_batches) * train_set_size)
    train_sampler = faw_batch_sampler(all_batches[:train_until_index])
    test_sampler = faw_batch_sampler(all_batches[train_until_index:])
    train_dl = DataLoader(ds, batch_sampler=train_sampler)
    test_dl = DataLoader(ds, batch_sampler=test_sampler)

##save test and train indices in file to be used later
    with open(os.path.join(outputs_dir,"test_indices"),'wb') as f:
        pickle.dump(all_batches[train_until_index:], f)
    with open(os.path.join(outputs_dir,"train indices"),'wb') as f:
        pickle.dump(all_batches[:train_until_index], f)
    #save test dl to be used in test
    with open(os.path.join(outputs_dir,"test dl"),'wb') as f:
        pickle.dump(test_dl, f)

    # the cnn
    model = FawNet()
    if is_gpu: model.cuda()
    train_epochs(model,train_dl,epochs,train_until_index,batch_size,with_pbar,print_loss)
    torch.save(model.state_dict(), os.path.join(outputs_dir,'faw_trained.pt'))

if __name__ == '__main__':
    main()
