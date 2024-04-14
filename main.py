import torch
import os
from torch.utils.data import DataLoader
import sys
from model import FakeNet
from data_loader import FakeDetectDataset


def train(train_load, model, criterion, optimizer, is_cuda=False):
    for (batch_no, sample) in enumerate(train_load):
        inp = sample[0]
        label = torch.tensor(sample[1], dtype=torch.float)
        if is_cuda:
            inp = inp.to('cuda')
            label = label.to('cuda')
        # label = label.cuda()
        optimizer.zero_grad()
        output = model(inp)
        print(output.squeeze())
        print(label)
        loss = criterion(output.squeeze(), label)
        loss.backward()
        optimizer.step()
        print("BATCH: " + str(batch_no), " loss --> " + str(loss))


def validate(val_loader, model, criterion, optimizer, is_cuda=False):
    print("#### VALIDATION")
    with torch.no_grad():
        for (label, sample) in enumerate(val_loader):
            inp = sample[0]
            label = torch.tensor(sample[1], dtype=torch.float)
            if is_cuda:
                inp = inp.to('cuda')
                label = label.to('cuda')

            output = model(inp)
            output = output.squeeze()
            print(output)
            print(label)
            class_pred = torch.sigmoid(output)
            class_pred[class_pred > 0.5] = 1
            class_pred[class_pred < 0.5] = 0
            acc = torch.sum(class_pred == label) / len(label)
            loss = criterion(output, label)
            print('Acc (VAL SET): ' + str(acc) + 'valid loss cross_ent: ' + str(loss))


def train_wrapper(is_cuda=False):
    torch.cuda.empty_cache()
    data_path = os.path.join(os.getcwd(), "fixed_images")

    momentum = 0.9
    lr = 0.01
    save_freq = 3
    model = FakeNet()
    model = torch.nn.DataParallel(model)
    if is_cuda:
        model = model.to('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_fnc = torch.nn.BCEWithLogitsLoss()
    if is_cuda:
        loss_fnc = loss_fnc.to('cuda')
    start_epoch = 0
    # TODO enable later
    from_checkpoint = False
    if from_checkpoint:
        check_p = os.listdir(".")
        check_p = max(list(map(lambda x: x[len('fake_classsifier_'): -1 * len('_checkpoint.pth.tar')], check_p)))
        load_name = os.path.join('/content', 'fake_classsifier_' + str(check_p) + '_checkpoint.pth.tar')
        loaded_model = torch.load(load_name)
        model.load_state_dict(loaded_model['state_dict'])
        optimizer.load_state_dict(loaded_model['optimizer'])
        start_epoch = loaded_model['epoch']

    dataset = FakeDetectDataset(root_path=data_path)
    data_train, data_val = torch.utils.data.random_split(dataset, [360, 39])
    print("dataset created")

    for epoch in range(start_epoch, 30):
        data_loader_params = {'dataset': data_train, 'batch_size': 20, 'shuffle': True, 'sampler': None,
                              'batch_sampler': None, 'num_workers': 0, 'collate_fn': None}
        train_loader = DataLoader(**data_loader_params)
        data_loader_params = {'dataset': data_val, 'batch_size': 39, 'shuffle': True, 'sampler': None,
                              'batch_sampler': None, 'num_workers': 0, 'collate_fn': None}
        test_loader = DataLoader(**data_loader_params)
        if epoch + 1 % save_freq + 1 == 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.5

        model.train()
        train(train_loader, model, loss_fnc, optimizer, is_cuda)
        model.eval()
        validate(test_loader, model, loss_fnc, optimizer, is_cuda)

        save_name = os.path.join('/content/drive/MyDrive', 'fake_class2_' + str(epoch) + '_checkpoint.pth.tar')
        torch.save({
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict(), }, save_name)
if __name__ == '__main__':
    train_wrapper()

