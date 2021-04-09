import argparse
import torch.optim
from torch.utils.data import DataLoader
from Data_Loader import make_ds_and_batches
import pickle
from services import faw_batch_sampler
import os

def main():
    ##### receiving user input
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_file", help="File path to the reports table")
    parser.add_argument("images_table_file", help="File path to the image table")
    parser.add_argument("images_root_directory", help="path to images root directory")
    parser.add_argument("outputs_directory", help="path to directory where outputs will be saved")
    parser.add_argument("model_path", help="path to model")
    parser.add_argument("test_indices_path", help="path to file with test indices")
    parser.add_argument("-bs", "--bad_shapes", help="File with paths to bad images", default=None)
    args = parser.parse_args()

    #######
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')


    #get the model and test indices
    model = torch.load(args.model_path, map_location=device)
    with open(args.test_indices_path, 'rb') as f:
        test_indices = pickle.load(f)
    test_sampler = faw_batch_sampler(test_indices)
    ds = make_ds_and_batches(args.reports_file,args.images_table_file,args.images_root_directory,'test',bad_shape_images_path=args.bad_shaoe_images_path)[0]
    test_dl = DataLoader(ds, batch_sampler=test_sampler)

    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    torch.no_grad()
    for i, data in enumerate(test_dl, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = outputs.flatten()
        assert ( labels.shape == outputs.shape), f"labels.shape = {labels.shape}, outputs.shape = {outputs.shape}"
        loss = criterion(outputs, labels.type(torch.float32))
        running_loss += loss.item()
        if i % 10 == 0 and i > 0:  # print every 100 mini-batches
            msg = f'current loss {running_loss / 10}'
            print(msg)
        running_loss = 0.0


if __name__ == '__main__':
    main()
