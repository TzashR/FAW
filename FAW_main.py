# %%
import argparse
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from CNN import FawNet
from Data_Loader import process_data, faw_batch_sampler, make_batches, FawDataset, faw_transform


def main():
    ### test if cuda is connected:
    # print(f' cuda  current device {torch.cuda.current_device()}')
    #
    # print(f' cuda device{torch.cuda.device(0)}')
    #
    # print(f' device count {torch.cuda.device_count()}')
    #
    # print(f' device name {torch.cuda.get_device_name(0)}')
    #
    # print(f' is cuda available {torch.cuda.is_available()}')
    ###

    # receiving user input
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_file", help="File path to the reports table")
    parser.add_argument("images_table_file", help="File path to the image table")
    parser.add_argument("images_root_directory", help="path to images root directory")
    parser.add_argument("--is_gpu", help="path to images root directory", type=bool, default=False)
    parser.add_argument("-pb","--with_pbar", help="Should have progress bar", type=bool, default=False)

    args = parser.parse_args()

    reports_file = args.reports_file
    images_file = args.images_table_file
    images_root_directory = args.images_root_directory
    is_gpu = args.is_gpu
    with_pbar = args.with_pbar

    train_set_size = 0.75  # how many (out of 1) should be in the training set.
    test_set_size = 0.25
    val_set_size = 1 - test_set_size - train_set_size
    batch_size = 1
    epochs = 2

    assert (1 > test_set_size > 0 and 1 > train_set_size > 0 and train_set_size + test_set_size + val_set_size == 1)

    ##get and process the data
    reports_df = pd.read_excel(reports_file, header=None)
    images_df = pd.read_excel(images_file, header=None)

    usable_reports_dic, usable_reports_lst, index_to_label, seedling_reports, missing_image_files, reportless_images, total_images = process_data(
        reports_df, images_df, images_root_directory)

    ds = FawDataset(images=usable_reports_lst, labels=index_to_label, transform=faw_transform)

    all_batches = make_batches(batch_size, usable_reports_dic)

    train_until_index = int(len(all_batches) * train_set_size)

    train_sampler = faw_batch_sampler(all_batches[:train_until_index])
    test_sampler = faw_batch_sampler(all_batches[train_until_index:])

    train_dl = DataLoader(ds, batch_sampler=train_sampler)
    test_dl = DataLoader(ds, batch_sampler=test_sampler)

    # the cnn
    model = FawNet()
    if is_gpu: model = model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    epoch_loss = 0.0
    epochs_bar = tqdm(epochs, total=epochs, disable=(not with_pbar), desc="epochs", position=0,leave=True)
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        batches_bar = tqdm(train_until_index//batch_size, total=train_until_index//batch_size, disable=(not with_pbar), desc="batches in epoch", position=0,leave=True)
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            labels = torch.FloatTensor(labels.float())
            if is_gpu: inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.flatten()
            assert (labels.shape == outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (with_pbar):
                batches_bar.update(n=1)
            # if i == 1 :  # print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0
            #     break
        # print(f' loss for epoch {epoch} = {epoch_loss / train_until_index}')
        if (with_pbar):
            epochs_bar.update(n=1)
    print("finished training!")
    torch.save(model.state_dict(), 'faw_trained.pt')


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        print(f'@@@@@@ {p.shape}')
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == '__main__':
    main()
