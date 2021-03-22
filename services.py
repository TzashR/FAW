import os
import random
from tqdm import tqdm
import torch.optim
from torch.utils.data import DataLoader

###DATA PROCESSING SERVICES
def fix_id(id, i=-1):
    # check_id(i, id)
    return (id + 50) // 100 * 100


def check_id(i, id):
    # print(i, id)
    id1 = (id // 100) * 100
    # print(id1)
    assert id == id1 or (id - 4) == id1 or (id - 8) == id1 or (id - 96) == id1 or (id - 92) == id1


def make_actual_file_path(measurement_path, root_folder):
    '''
    @param measurement_path: A file path from photo conversion excel
    @param root_folder: Folder containing images
    @return: path to actual file in the folder
    '''

    splitted_measurement_path = measurement_path.split("/")
    actual_path = os.path.join(root_folder, splitted_measurement_path[2], splitted_measurement_path[3])

    return actual_path


# TODO I need to get the index for each image
def process_data(reports_df, images_df, images_dir, bad_shaped_images=None):
    '''
    processes that data so that it can be used in data loader

    @param reports_table:
    @param images_table:
    @param images_dir:
    @bad_shaped_images: a set of file paths to images with bad shape. They will be ignored
    @return: -
        usable : { report_id: (infest_level, [image_file_path])}
        seedling_reports : {report id: [image_file_path}
        number of missing images (in reports but not in image dir)
        number of reportless images (in dir but not in reoprts)
        number of total images in dir


    '''
    reports_dic = {}
    seedlings_dic = {}

    # creates dictionary for reports (one for usable  and on for seedlings)
    for i in range(1, len(reports_df)):
        report_id = fix_id(fix_id(reports_df[0][i]))
        if reports_df[9][i] == 'Seedling':
            seedlings_dic[report_id] = []
        else:
            reports_dic[report_id] = {'infest_level': reports_df[7][i], 'images': []}

    num_images_not_in_reports = 0
    num_missing_image_files = 0
    total_images = 0
    usable_index = 0
    usable_reports_lst = []
    index_to_label = {}

    # assigns image files to their respective report in the dictionaries
    for i in range(1, len(images_df)):
        measurement_id = fix_id(images_df[0][i])
        image_file_path = make_actual_file_path(images_df[1][i], images_dir)

        if not os.path.isfile(image_file_path):
            num_missing_image_files += 1
            continue
        elif bad_shaped_images is not None and image_file_path in bad_shaped_images:
            continue
        if measurement_id in reports_dic:
            usable_reports_lst.append(image_file_path)
            reports_dic[measurement_id]['images'].append((image_file_path, usable_index))
            index_to_label[usable_index] = reports_dic[measurement_id]['infest_level']
            usable_index += 1
        elif measurement_id in seedlings_dic:
            seedlings_dic[measurement_id].append(image_file_path)
        else:
            num_images_not_in_reports += 1
        total_images += 1
    return (reports_dic, usable_reports_lst, index_to_label, seedlings_dic,
            num_missing_image_files, num_images_not_in_reports, total_images)


def make_batches(batch_size, reports_dic, min_images=5, seed=0):
    # list of all reports with at least min_images images
    usable_reports = list(filter(lambda report: len(report[1]['images']) >= min_images, reports_dic.items()))
    random.shuffle(usable_reports)
    n = len(usable_reports)
    num_of_batches, remainder = n // batch_size, n % batch_size

    # batches at this point are reports, we need to convert them to image indices
    batch_reports = [usable_reports[i:i + batch_size] for i in range(num_of_batches)]
    if remainder > 0: batch_reports.append(usable_reports[-remainder:])

    batches = []
    # converts all reports in batches to a list of image indices
    for rep_batch in batch_reports:
        batch_indices = []
        for report in rep_batch:
            for image_tuple in report[1]['images']:  # image tuple contains path, index
                batch_indices.append(image_tuple[1])
        batches.append(batch_indices)
    return batches


def faw_batch_sampler(batches):
    '''
    creates the batch sampler that the model uses
    @param batches: batches generated by make_batches
    @return:
    '''
    for i in range(len(batches)):
        yield batches[i]

### TRAINING AND TESTING SERVICES

def get_n_params(model):
    '''
    prints the number of parameters in a model
    @param model:
    @return:
    '''
    pp = 0
    for p in list(model.parameters()):
        print(f'@@@@@@ {p.shape}')
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp

def is_cuda():
    ### test if cuda is connected:
    print(f' cuda  current device {torch.cuda.current_device()}')
    print(f' cuda device{torch.cuda.device(0)}')
    print(f' device count {torch.cuda.device_count()}')
    print(f' device name {torch.cuda.get_device_name(0)}')
    print(f' is cuda available {torch.cuda.is_available()}')
    ###

def train_epochs(model,train_dl,num_epochs,train_until_index,batch_size, is_gpu, with_pbar = False, print_loss = False):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    epoch_loss = 0.0
    epochs_bar = tqdm(num_epochs, total=num_epochs, disable=(not with_pbar), desc="epochs", position=0, leave=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        batches_bar = tqdm(train_until_index//batch_size, total=train_until_index//batch_size, disable=(not with_pbar), desc="batches in epoch", position=1,leave=True)
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data
            if is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.flatten()
            assert (labels.shape == outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if with_pbar:
                batches_bar.update(n=1)
            if print_loss:
                if i%100 == 0 :  # print every 100 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0
                        break
                print(f' loss for epoch {epoch} = {epoch_loss / train_until_index}')
        if with_pbar:
            epochs_bar.update(n=1)
    print("finished training!")
