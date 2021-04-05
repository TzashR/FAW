import argparse
import os
import pickle

import torch.optim
from torch.utils.data import DataLoader

from CNN import FawNet
from Data_Loader import make_ds_and_batches
from services import faw_batch_sampler
from train_and_test import train_epoch


def main():
    ##### receiving user input
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", help="how many epochs", type=int)
    parser.add_argument("batch_size", help="batch size", type=int)
    parser.add_argument("reports_file", help="File path to the reports table")
    parser.add_argument("images_table_file", help="File path to the image table")
    parser.add_argument("images_root_directory", help="path to directory where images are stored")
    parser.add_argument("outputs_dir", help="path directory where outputs will be saved")
    parser.add_argument("--with_gpu", help="path to images root directory", action='store_true')
    parser.add_argument("-pb", "--with_pbar", help="Should have progress bar", action='store_true')
    parser.add_argument("-pl", "--print_loss", help="Should print loss while training", action='store_true')
    parser.add_argument("-bs", "--bad_shaped_images", help="File with paths to bad images", default=None)
    args = parser.parse_args()

    ds, all_batches = make_ds_and_batches(args.reports_file, args.images_table_file, args.images_root_directory,
                                          'train',
                                          bad_shape_images_path=args.bad_shaped_images, batch_size=args.batch_size)

    train_set_size = 0.75  # how many (out of 1) should be in the training set.
    test_set_size = 0.25
    val_set_size = 1 - test_set_size - train_set_size
    assert (1 > test_set_size > 0 and 1 > train_set_size > 0 and train_set_size + test_set_size + val_set_size == 1)

    train_until_index = int(len(all_batches) * train_set_size)

    ##save test and train indices in file to be used later
    with open(os.path.join(args.outputs_dir, "test_indices"), 'wb') as f:
        pickle.dump(all_batches[train_until_index:], f)
    with open(os.path.join(args.outputs_dir, "train indices"), 'wb') as f:
        pickle.dump(all_batches[:train_until_index], f)

    # the cnn
    model = FawNet()
    if args.with_gpu: model.cuda()
    ## train the model
    for epoch in range(args.epochs):
        train_sampler = faw_batch_sampler(all_batches[:train_until_index])
        train_dl = DataLoader(ds, batch_sampler=train_sampler)

        if args.with_pbar: print(f'epoch #{epoch + 1}')
        epoch_loss = train_epoch(model=model, train_dl=train_dl, train_until_index=train_until_index,
                                  batch_size=args.batch_size, with_pbar=args.with_pbar, print_loss=args.print_loss,
                                  is_gpu=args.is_gpu)
        if args.print_loss:
            print(f' loss for epoch {epoch} = {epoch_loss / train_until_index}')
    print("finished training!")

    ##save the trained model
    torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'faw_trained.pt'))
    # test the model
    # test_sampler = faw_batch_sampler(all_batches[train_until_index:])
    # test_dl = DataLoader(ds, batch_sampler=test_sampler)
    # test_model(model,test_dl, args.outputs_dir)


if __name__ == '__main__':
    main()

# #%%
# from train_and_test import train_epochs
# reports_file = "D:\Kenya IPM field reports.xls"
# images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
# bad_shape_images = r"C:\Users\User\Development\MA\FAW\bad shaped images clean"
# USB_PATH = r"D:\2019_clean2"
#
# ds,all_batches = make_ds_and_batches(reports_file,images_file,USB_PATH,'train',bad_shape_images_path=bad_shape_images,batch_size=1)
#
# #%%
# train_set_size = 0.75  # how many (out of 1) should be in the training set.
# test_set_size = 0.25
# val_set_size = 1 - test_set_size - train_set_size
# assert (1 > test_set_size > 0 and 1 > train_set_size > 0 and train_set_size + test_set_size + val_set_size == 1)
#
# train_until_index = int(len(all_batches) * train_set_size)
# train_sampler = faw_batch_sampler(all_batches[:train_until_index])
# test_sampler = faw_batch_sampler(all_batches[train_until_index:])
# train_dl = DataLoader(ds, batch_sampler=train_sampler)
# test_dl = DataLoader(ds, batch_sampler=test_sampler)
#
# #%%
# model = FawNet()
# train_epochs(model=model, train_dl=train_dl, num_epochs=2, train_until_index = train_until_index,
#              batch_size=1, with_pbar=True, print_loss=True, is_gpu=False)
