import torch.optim
from tqdm import tqdm


def train_epoch(model, train_dl, train_until_index, batch_size, device, optimizer, criterion, with_pbar=False,
                print_loss=False):
    epoch_loss = 0.0
    running_loss = 0.0
    batches_bar = tqdm(train_until_index // batch_size, total=train_until_index // batch_size, disable=(not with_pbar),
                       desc="batches in epoch", position=0, leave=True, ascii=True)
    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
        print(f' inputs.shape = {inputs.shape}, labels.shape = {labels.shape}')

        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a  # free inside reserved

        print(f'total memory = {t} , reserved memory = {r}, allocated memory = {a}, free memory = {f}')

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.flatten()
        assert (labels.shape == outputs.shape), f"labels.shape = {labels.shape}, outputs.shape = {outputs.shape}"
        loss = criterion(outputs, labels.type(torch.float32))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if with_pbar:
            batches_bar.update(n=1)
        if print_loss:
            if i % 100 == 0 and i > 0:  # print every 100 mini-batches
                msg = f'current loss {running_loss / 100}'
                if with_pbar:
                    tqdm.write(msg)
                else:
                    print(msg)
                running_loss = 0.0

    return epoch_loss
