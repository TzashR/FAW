import torch.optim
from tqdm import tqdm

def train_epochs(model,train_dl,num_epochs,train_until_index,batch_size, is_gpu = False, with_pbar = False, print_loss = False):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    epoch_loss = 0.0
    for epoch in range(num_epochs):
        if with_pbar: tqdm.write(f'epoch #{epoch+1}')
        running_loss = 0.0
        batches_bar = tqdm(train_until_index//batch_size, total=train_until_index//batch_size, disable=(not with_pbar), desc="batches in epoch", position=0,leave=True, ascii =True)
        for i, data in enumerate(train_dl, 0):
            inputs, labels = data

            if is_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.flatten()
            assert (labels.shape == outputs.shape)
            loss = criterion(outputs, labels.type(torch.float32))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if with_pbar:
                batches_bar.update(n=1)
            if print_loss:
                msg = f'current loss {running_loss/10}'
                if i%10 == 0 and i>0 :  # print every 100 mini-batches
                    if with_pbar:
                        tqdm.write(msg)
                    else: print(msg)
                    running_loss = 0.0
        if print_loss:
            msg = f' loss for epoch {epoch} = {epoch_loss / train_until_index}'
    print("finished training!")

def test_model(model,test_dl,outputs_dir):
    '''
    Tests a train model
    @param model: trained model
    @param outputs_dir: dir to save output in
    @return:
    '''
    pass
