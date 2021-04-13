import argparse
import os
import pickle
from datetime import datetime

import torch.optim
from torch.utils.data import DataLoader

from CNN import FawNet
from Data_Loader import make_ds_and_batches
from services import faw_batch_sampler
from services import is_cuda
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
    assert train_until_index > 0

    outputs_dir = os.path.join(args.outputs_dir,f"{datetime.today().strftime('%d%m%Y')} {datetime.now().strftime('%H%M')}")
    os.mkdir(outputs_dir)
    os.chmod(outputs_dir,0o777)
    ##save test and train indices in file to be used later
    with open(os.path.join(outputs_dir, "test_indices"), 'wb') as f:
        pickle.dump(all_batches[train_until_index:], f)
    with open(os.path.join(outputs_dir, "train indices"), 'wb') as f:
        pickle.dump(all_batches[:train_until_index], f)

    # the cnn
    with_gpu =torch.cuda.is_available()

    if with_gpu:
        device = torch.device("cuda:0")
        print("running on GPU")
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved
        print(f'total memory = {t} , reserved memory = {r}, allocated memory = {a}, free memory = {f}')

    else:
        device = torch.device("cpu")
        print("running on CPU")


    model = FawNet().to(device)

    ## train the model
    print("on our way")
    for epoch in range(args.epochs):
        train_sampler = faw_batch_sampler(all_batches[:train_until_index])
        train_dl = DataLoader(ds, batch_sampler=train_sampler)

        if args.with_pbar: print(f'epoch #{epoch + 1}')
        epoch_loss = train_epoch(model=model, train_dl=train_dl, train_until_index=train_until_index,
                                 batch_size=args.batch_size, with_pbar=args.with_pbar, print_loss=args.print_loss,
                                 device = device)
        if args.print_loss:
            print(f' loss for epoch {epoch} = {epoch_loss / train_until_index}')
    print("finished training!")

    ##save the trained model
    torch.save(model.state_dict(), os.path.join(outputs_dir, 'faw_trained.pt'))
    # test the model
    # test_sampler = faw_batch_sampler(all_batches[train_until_index:])
    # test_dl = DataLoader(ds, batch_sampler=test_sampler)
    # test_model(model,test_dl, args.outputs_dir)


if __name__ == '__main__':
    main()

# #%%
# reports_file = "D:\Kenya IPM field reports.xls"
# images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
# bad_shape_images = r"C:\Users\User\Development\MA\FAW\bad shaped images clean"
# USB_PATH = r"D:\2019_clean2"
#
# ds,all_batches = make_ds_and_batches(reports_file,images_file,USB_PATH,'train',bad_shape_images_path=bad_shape_images,batch_size=1)
#
# #%%
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
# #
# #%%
# myModel = FawNet()
# model = torch.load(r"C:\Users\User\Development\MA\FAW\outputs\08042021 1231\faw_trained.pt",map_location=torch.device('cpu'))
# myModel.load_state_dict(model)
#
# #%%
# criterion = torch.nn.MSELoss()
# running_loss = 0.0
# torch.no_grad()
# x = enumerate(train_dl, 0)
# #%%
# inputs, labels = x.__next__()[1]
# outputs = myModel(inputs)
# outputs = outputs.flatten()
# assert (labels.shape == outputs.shape), f"labels.shape = {labels.shape}, outputs.shape = {outputs.shape}"
# loss = criterion(outputs, labels.type(torch.float32))
#
