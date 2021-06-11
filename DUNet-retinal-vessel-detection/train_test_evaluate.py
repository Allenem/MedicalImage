import os
import torch
import argparse
import numpy as np
import PIL.Image as Image
from torch import nn, optim
from torch._C import device
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from models import Unet, Unet_plus_plus
from loaddata import LoadDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
y_transforms = transforms.ToTensor()


def Dice(y_true, y_pred):
    # input is array
    if np.max(y_true) > 1:
        y_true = y_true / 255.
    if np.max(y_pred) > 1:
        y_pred = y_pred / 255.
    return 2 * np.sum(y_true * y_pred) / (np.sum(y_pred) + np.sum(y_true))


def train_model(model, criterion, optimizer, dataloader, args, num_epochs=5):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dt_size = len(dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(type(outputs))
            # print(outputs.shape)
            # print(outputs)
            # print(type(labels))
            # print(labels.shape)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print('%d/%d,train_loss:%0.3f' %
                  (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        print('epoch %d loss:%0.3f' % (epoch+1, epoch_loss/step))
        torch.save(model.state_dict(), 'weights_{}_{}.pth'.format(
            args.datasetname, (epoch+1)))
    return model


def train(args):
    model = Unet_plus_plus(1, 1).to(device)
    batch_size = args.batch_size
    # BCELoss = − 1/n ∑ ( y_n × ln ⁡ x_n + ( 1 − y_n ) × ln ⁡ ( 1 − x_n ) )
    # BCEWithLogitsLoss 就是把 Sigmoid 和 BCELoss 合成一步
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    load_dataset = LoadDataset(
        'data_crop/'+args.datasetname+'/train/', transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(
        load_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, args)


def test(args):
    model = Unet_plus_plus(1, 1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    batch_size = 1
    load_dataset = LoadDataset(
        'data_crop/'+args.datasetname+'/test/', transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(load_dataset, batch_size=batch_size)
    model.eval()
    Dice_score_all = 0
    idx = 0
    threshold = 0.5
    for x, y_true in dataloaders:
        idx += 1
        y_pred = model(x).sigmoid()
        y_pred = y_pred.detach().numpy().reshape((256, 256))
        # print(max([item for sublist in y_pred for item in sublist]))
        # print(min([item for sublist in y_pred for item in sublist]))
        y_pred = np.int64(y_pred > threshold)
        # print(max([item for sublist in y_pred for item in sublist]))
        # print(min([item for sublist in y_pred for item in sublist]))
        y_true = y_true.detach().numpy().reshape((256, 256))
        # print(y_pred)
        # print(type(y_pred))
        # print(y_pred.shape)
        # print(y_true)
        # print(type(y_true))
        # print(y_true.shape)
        Dice_score = Dice(y_true, y_pred)
        Dice_score_all += Dice_score
        print('index:'+str(idx)+' dice score:'+str(Dice_score))
        # y_pred = torch.squeeze(y_pred).numpy()
        y_pred_img = Image.fromarray(np.uint8(y_pred*255))
        if y_pred_img.mode == "F":
            y_pred_img = y_pred_img.convert('1')
        if not os.path.exists('result/'+args.datasetname + '/test/'):
            os.makedirs('result/'+args.datasetname + '/test/')
        y_pred_img.save('result/'+args.datasetname +
                        '/test/%03d_predict.png' % idx)
    Dice_score_average = Dice_score_all/(len(dataloaders.dataset)*batch_size)
    print('Average Dice Score: {:.4f}%'.format(Dice_score_average*100))


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument('action', type=str,
                       help='train or test', default='train')
    parse.add_argument('--batch_size', type=int, default=8)
    parse.add_argument('--ckpt', type=str,
                       help='the path of model weight file', default='weights_CHASE_5.pth')
    parse.add_argument('--datasetname', type=str,
                       help='CHASE or DRIVE', default='CHASE')
    args = parse.parse_args()

    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        test(args)
