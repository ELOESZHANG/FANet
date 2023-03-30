# coding=UTF-8
import torch
import sys
import os
import torch.nn.functional as F
sys.path.append('/home/xmn/PycharmProjects/SOD/libs')
sys.path.append(os.pardir)
from datasets import rgbd_transforms
from testnet import Imagemodel
from datasets import rgbd_datasets
from tqdm import tqdm
from PIL import Image
# import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
import os
from torchvision.transforms import functional as tf
from utils.pyt_utils import load_model
import time
import numpy as np
import torch.nn as nn


bce_loss = nn.BCELoss()
def td_loss(pred,target):
    output_loss = bce_loss(pred, target.float())
    return output_loss

cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if cuda else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()  # return index
    print("Running on", torch.cuda.get_device_name(current_device))  # ,torch.cuda.device_count())
else:
    print("Running on CPU")

data_transforms = rgbd_transforms(
    input_size=(256, 256),
    output_size=(256, 256),  # (56,56)
    image_mode=False
)

val_dataset = rgbd_datasets(
    name_list=["train_test4", "train_test3"],
    split_list=["val", "val"],
    config_path='/home/xmn/PycharmProjects/myNetwork/config/datasets.yaml',
    root='/home/xmn/PycharmProjects/myNetwork/data/datasets',
    training=False,
    transforms=data_transforms['val'],
    read_clip=True,
    random_reverse_clip=False,
    clip_len=3
)

val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=3,
    num_workers=4,
    shuffle=False,
    drop_last=False
)

dataloaders = {'val': val_dataloader}

model = Imagemodel()
path = '/home/xmn/PycharmProjects/new/models/checkpoints/video_test_current_newloss3.pth'  # video_epoch-14.30.94708843231201
model.load_state_dict(torch.load(path), strict=False)
# for key in enumerate(torch.load(path)):
#     print(key)

model.to(device=1)
model.eval()
unloader = torchvision.transforms.ToPILImage()

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def test():
    for data in tqdm(dataloaders['val']):
        images = [frame['image'].to(device=1) for frame in data]
        depth_maps = [frame['depth'].to(device=1) for frame in data]
        with torch.no_grad():
            time_start = time.time()
            preds = model(images, depth_maps)
            preds = [torch.sigmoid(pred) for pred in preds]  ##test here
            time_end = time.time()
            six_frames_time = time_end - time_start

        for i, pred_ in enumerate(preds):
            for j, pred in enumerate(pred_.detach().cpu()):
                dataset = data[i]['dataset'][j]
                image_id = data[i]['depth_id'][j]
                height = data[i]['height'].numpy()[j]
                width = data[i]['width'].numpy()[j]
                result_path = os.path.join("/home/xmn/PycharmProjects/myNetwork/test/{}/{}.png".format(dataset, image_id))
                result = tf.to_pil_image(pred)
                res = result.resize((height, width))
                dirname = os.path.dirname(result_path)  # get

                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                res.save(result_path)

if __name__ == "__main__":
    test()