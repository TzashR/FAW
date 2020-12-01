#%%
import pandas as pd
import torch.optim
from torch.utils.data import DataLoader

from CNN import FawNet
from Data_Loader import process_data, faw_batch_sampler, make_batches, FawDataset, faw_transform

def main():
    # user defined variables:
    reports_file = "D:\Kenya IPM field reports.xls"
    images_file = "D:\Kenya IPM measurement to photo conversion table.xls"
    images_root_directory = r"D:\2019_clean2"
    train_set_size = 0.75  # how many (out of 1) should be in the training set.
    test_set_size = 0.25
    val_set_size = 1 - test_set_size - train_set_size
    batch_size = 20

    val_set_size = 1 - train_set_size-test_set_size
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    for epoch in range(2):
        runing_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("finished training!")

if __name__ == '__main__':
    main()



