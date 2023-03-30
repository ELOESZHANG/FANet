import torch
import torch.nn as nn
from torch import optim
import sys
sys.path.append('/home/xmn/PycharmProjects/myNetwork/libs')
import os
import torch.nn.functional as F
#from m utils.pyt_utils import load_model
from torch.utils import data
import numpy as np
from utils.metric import StructureMeasure, Eval_Fmeasure, Eval_Fbw_measure
from tqdm import tqdm
from datasets import rgbd_transforms
from datasets import rgbd_datasets
# from testnet import VGG16_deconv
from visdom import Visdom
from testnet import Imagemodel

viz = Visdom(port=12345, env="test3")
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device() #return index
    print("Running on", torch.cuda.get_device_name(current_device))#,torch.cuda.device_count())
else:
    print("Running on CPU")


bce_loss = nn.BCEWithLogitsLoss()
def td_loss(pred, label):#pred2,,depth,depthtarget,
    output_loss = bce_loss(pred, label.float())
    return output_loss



data_transforms = rgbd_transforms(
    input_size=(256, 256),
    output_size=(256, 256),
    image_mode=False
)


train_dataset = rgbd_datasets(
    name_list=["train_test4", "train_test3"],
    split_list=["train", "train"],
    config_path='/home/xmn/PycharmProjects/myNetwork/config/datasets.yaml',#位置
    root='/home/xmn/PycharmProjects/myNetwork/data/datasets',
    training=True,
    transforms=data_transforms['train'],
    read_clip=True,
    random_reverse_clip=False,
    clip_len=3
)

val_dataset = rgbd_datasets(
    name_list=["train_test4", "train_test3"],
    split_list=["val", "val"],
    config_path='/home/xmn/PycharmProjects/myNetwork/config/datasets.yaml',#位置
    root='/home/xmn/PycharmProjects/myNetwork/data/datasets',
    training=False,
    transforms=data_transforms['val'],
    read_clip=True,
    random_reverse_clip=False,
    clip_len=3
)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=3,
    num_workers=4,
    shuffle=True,
    drop_last=True
)

val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=3,
    num_workers=4,
    shuffle=False,
    drop_last=True
)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}

model = Imagemodel().to(device=1)
total_params = sum(p.numel() for p in model.parameters())
print('总参数量2：{}'.format(total_params))
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4, momentum=0.9)
if not os.path.exists('/home/xmn/PycharmProjects/myNetwork/models/checkpoints'):#位置
    os.makedirs('/home/xmn/PycharmProjects/myNetwork/models/checkpoints')

viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
viz.line([0.], [0.], win='acc', opts=dict(title='Sm_acc'))
viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))

def train():
    best_smeasure = 0.0
    best_epoch = 0

    for epoch in range(0, 100):  # 位置
        phases = ['train', 'val']
        for phase in phases:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0

            running_mae = 0.0
            running_smean = 0.0
            print("{} epoch {}...".format(phase, epoch))
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                imgs, depth_maps, labels, labels1, labels2, labels3, frameids = [], [], [], [], [], [], []
                for frame in data:
                    depth_maps.append(frame['depth'].to(device=1))
                    imgs.append(frame['image'].to(device=1))
                    labels.append(frame['label'].to(device=1))
                    frameids.append(frame['depth_id'])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(imgs, depth_maps)
                    loss = []
                    for pred, label, frameid in zip(preds, labels, frameids):
                        _loss = td_loss(pred, label)
                        loss.append(_loss)

                    if phase == 'train':
                        torch.autograd.backward(loss)  # ,retain_graph=True
                        optimizer.step()

                for _loss in loss:
                    running_loss += _loss.item()
                preds = [torch.sigmoid(pred) for pred in preds]  # activation

                # iterate list
                for i, (label_, pred_) in enumerate(zip(labels, preds)):
                    # iterate batch
                    for j, (label, pred) in enumerate(zip(label_.detach().cpu(), pred_.detach().cpu())):
                        pred_idx = pred[0, :, :].numpy()  # pred_idx = pred[0,:,:].numpy()
                        label_idx = label[0, :, :].numpy()
                        if phase == 'val':
                            running_smean += StructureMeasure(pred_idx.astype(np.float32),
                                                              (label_idx >= 0.5).astype(np.bool))

                        running_mae += np.abs(pred_idx - label_idx).mean()

            samples_num = len(dataloaders[phase].dataset)
            print(samples_num)
            samples_num *= 3
            epoch_loss = running_loss / samples_num
            epoch_mae = running_mae / samples_num
            if phase == 'train':
                viz.line([epoch_loss], [epoch], win='train_loss', name='loss1', update='append')
            if phase == 'val':
                viz.line([epoch_loss], [epoch], win='val_loss', name='loss2', update='append')
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} MAE: {:.4f}'.format(phase, epoch_mae))

            # save current best
            if phase == 'val':
                epoch_smeasure = running_smean / samples_num
                viz.line([epoch_smeasure], [epoch], win='acc', name='acc1', update='append')
                print('{} S-measure: {:.4f}'.format(phase, epoch_smeasure))
                if epoch_smeasure > best_smeasure:
                    best_smeasure = epoch_smeasure
                    best_epoch = epoch
                    model_path = os.path.join('/home/xmn/PycharmProjects/myNetwork/models/check',
                                              "RGBD2.pth")
                    print("Saving current best Sm model at: {}".format(model_path))
                    torch.save(
                        model.state_dict(),
                        model_path,
                    )
        if epoch > 0 and epoch % 1 == 0:
            # save model
            model_path = os.path.join('/home/xmn/PycharmProjects/myNetwork/models/check/test/',
                                      "RGBD_test2.pth")  # "video_epoch-{}.{}.pth".format(epoch,epoch_loss)
            print("Backup model at: {}".format(model_path))
            torch.save(
                model.state_dict(),
                model_path,
            )

    print('Best S-measure: {} at epoch {}'.format(best_smeasure, best_epoch))

if __name__ == "__main__":
    train()