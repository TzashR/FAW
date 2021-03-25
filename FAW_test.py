import argparse
import pickle
import pandas as pd
import argparse
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from CNN import FawNet
from Data_Loader import  FawDataset, faw_transform, make_ds_and_batches
import pickle
from services import process_data, faw_batch_sampler, make_batches, train_epochs
import os

def main():
    ##### receiving user input
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_file", help="File path to the reports table")
    parser.add_argument("images_table_file", help="File path to the image table")
    parser.add_argument("images_root_directory", help="path to images root directory")
    parser.add_argument("outputs_directory", help="path to directory where outputs will be saved")
    parser.add_argument("test indices path", help="path to file with test indices")
    parser.add_argument("--is_gpu", help="path to images root directory", type=bool, default=False)
    parser.add_argument("-pb", "--with_pbar", help="Should have progress bar", type=bool, default=False)
    parser.add_argument("-pl", "--print_loss", help="Should print loss while training", default=False)
    parser.add_argument("-bs", "--bad_shapes", help="File with paths to bad images", default=None)
    args = parser.parse_args()

    #######

    ds = make_ds_and_batches(args.reports_file,args.images_file,args.images_root_directory,'test',bad_shape_images_path=args.bad_shaoe_images_path)
    #get the test indices
    with open(r"C:\Users\User\Development\MA\FAW\outputs\test_indices", 'rb') as f:
        test = pickle.load(f)

if __name__ == '__main__':
    main()
