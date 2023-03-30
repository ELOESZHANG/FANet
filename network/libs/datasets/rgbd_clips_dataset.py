# coding: utf-8

from __future__ import print_function

import os
import random
import yaml
import numpy as np
from glob import glob
from PIL import Image

import torch
from torch.utils import data

class SaliencyDataset(data.Dataset):#單幀

    def __init__(self, name,image_ext, label_ext,depth_ext,image_dir,
            label_dir,depth_dir, root, split, training, transforms):# 

        self.name = name
        self.root = os.path.join(root, name)
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.depth_ext = depth_ext
        self.image_dir = os.path.join(self.root, image_dir)
        self.label_dir = os.path.join(self.root, label_dir)
        self.depth_dir = os.path.join(self.root, depth_dir)

        self.split = split
        # not labels in inference mode
        self.training = training
        self.transforms = transforms
        self.files = []

    def _get_frame(self, frame_info):
        image_path = frame_info['image_path']
        depth_path = frame_info['depth_path']
        image = Image.open(image_path).convert('RGB') # RGB format
        depth = Image.open(depth_path).convert('RGB')
        #print(depth)
        image_size = image.size[:2]
        #depth_size = depth.size[:2]
        item = {'dataset': self.name,
                'depth_id': frame_info['depth_id'],
                'height': image_size[0],
                'width': image_size[1]}

        if 'label_path' in frame_info:
            
            label = np.array(Image.open(frame_info['label_path']).convert('1')).astype(np.int32)##convert('1')
            #if label.max() > 1:
            #    label = (label >= 128).astype(np.uint8) # convert 255 to 1
            #else:
            #    label = (label >= 0.5).astype(np.uint8)
            if label.max() > 1:
                label = label / 255
            label = Image.fromarray(label)
        else:
            label = None

        sample = {'image': image,'label': label,'depth': depth}#
        #print(sample['depth'])
        #sample = self.transforms(sample)
        #print(sample)
        item['image'] = sample['image']
        item['depth'] = sample['depth']
        if label is not None:
            item['label'] = sample['label']
        return item
    def _set_files(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.files)

class VideoDataset(SaliencyDataset):

    def __init__(self, video_split, default_label_interval, label_interval, frame_between_label_num=0, **kwargs):
        super(VideoDataset, self).__init__(**kwargs)

        self.video_split = video_split
        self.label_interval = label_interval
        self.frame_between_label_num = frame_between_label_num  # the number of frame without label between two frames with label
        self.default_label_interval = default_label_interval   # default labels interval of this dataset
        if self.frame_between_label_num >= self.label_interval * self.default_label_interval:
            raise ValueError("The number of frame without label {} should be smaller than {}*{}",
                                self.frame_between_label_num, self.label_interval, self.default_label_interval)

    def _get_frame_list(self, video):
        image_path_root = os.path.join(self.image_dir, video)
        depth_path_root = os.path.join(self.depth_dir, video)
        label_path_root = os.path.join(self.label_dir, video)
        # the list of all frame
        frame_list = sorted(glob(os.path.join(image_path_root, "*" + self.image_ext)))
        depth_list = sorted(glob(os.path.join(depth_path_root, "*" + self.depth_ext)))
        if not frame_list:
            raise FileNotFoundError(image_path_root)
        if not depth_list:
            raise FileNotFoundError(depth_path_root)
        #frame_id_list = [f.split("/")[-1].replace(self.image_ext, "") for f in frame_list]
        depth_id_list = [f.split("/")[-1].replace(self.depth_ext, "") for f in depth_list]
        frame_id_list = [f.split("/")[-1].replace(self.image_ext, "") for f in frame_list]
        frame_id_index = [depth_id_list.index(frame_id) for frame_id in frame_id_list]
        #depth_id_index = [depth_id_list.index(depth_id) for depth_id in depth_id_list]
        # the list of frame with labels
        label_list = sorted(glob(os.path.join(label_path_root, "*" + self.label_ext)))
        #if not label_list:
        #    raise FileNotFoundError(label_path_root)
        label_list = label_list[::self.label_interval] if self.training else label_list
        label_id_list = [f.split("/")[-1].replace(self.label_ext, "") for f in label_list]
        # the index of frames with label
        label_id_index = [depth_id_list.index(label_id) for label_id in label_id_list]

        return label_id_index,depth_id_list,frame_id_index, label_path_root, depth_path_root,image_path_root#
    
    def _get_video_info(self, video):
        label_id_index, depth_id_list,frame_id_index, label_path_root, depth_path_root,image_path_root= self._get_frame_list(video) #
        # set up video info 
        video_info = []
        for depth_id in depth_id_list:
            depth_path = os.path.join(depth_path_root, depth_id + self.depth_ext)
            frame_info = {'depth_id': "{}/{}".format(video, depth_id),
                        'depth_path': depth_path}
            video_info.append(frame_info)
        for index in frame_id_index:
            image_id = depth_id_list[index]
            image_path = os.path.join(image_path_root, image_id + self.image_ext)
            video_info[index]['image_path'] = image_path
        for index in label_id_index:
            image_id = depth_id_list[index]
            label_path = os.path.join(label_path_root, image_id + self.label_ext)
            video_info[index]['label_path'] = label_path
        return video_info, label_id_index

class VideoClipDataset(VideoDataset):
    
    def __init__(self, clip_len, random_reverse_clip, **kwargs):
        super(VideoClipDataset, self).__init__(**kwargs)

        self.random_reverse_clip = random_reverse_clip
        self.clip_len = clip_len if self.frame_between_label_num == 0 else self.frame_between_label_num + 2

        self.clips = []
        self.frame_wo_label_interval = (self.label_interval * self.default_label_interval) // (self.frame_between_label_num + 1)
        self._set_files()
    def _get_clips(self, video_index, label_id_index):
        indexes = []
        for index in label_id_index[:-1]:#有標籤幀的索引，除最後一幀，爲了加上沒標籤的幀
            indexes.append(index)
            for j in range(self.frame_between_label_num):#讓沒標籤的幀加到有標籤幀的中間
                indexes.append(index + self.frame_wo_label_interval * (j+1))
        indexes.append(label_id_index[-1])
        if len(indexes) < self.clip_len:
            indexes = indexes + [indexes[-1]] * (self.clip_len - len(indexes))
        clips = []
        clip_start_index = 0
        while clip_start_index <= len(indexes) - self.clip_len:
            clips.append({'video_index': video_index, 'clip_frame_index': indexes[clip_start_index:clip_start_index+self.clip_len]})
            clip_start_index += self.clip_len#-1 if self.training else self.clip_len#！！！！！！！
        # last clip
        if clip_start_index < len(indexes):
            clips.append({'video_index': video_index, 'clip_frame_index': indexes[len(indexes)-self.clip_len:len(indexes)]})
        return clips#一個list,每個元素是一個片段索引構成的字典，包括視頻片段的索引和每個片段中視頻幀的索引

    def _reset_files(self, clip_len, label_dir):
        self.files.clear()
        self.clips.clear()
        self.label_dir = label_dir
        self.label_interval = 1
        self.clip_len = clip_len
        self.frame_between_label_num = 0
        self._set_files()

    def _set_files(self):
        if self.split in list(self.video_split.keys()):
            for video_index, video in enumerate(self.video_split[self.split]):
                video_info, label_id_index = self._get_video_info(video)
                if not len(video_info):
                    continue
                self.files.append(video_info)
                self.clips += self._get_clips(video_index, label_id_index)
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def __getitem__(self, index):
        clip,clip_ = [],[]
        clip_frame_index = self.clips[index]['clip_frame_index']
        video_index = self.clips[index]['video_index']
        # random revese video when training
        if self.random_reverse_clip and random.randint(0,1):
            clip_frame_index = clip_frame_index[::-1]
        for i in clip_frame_index:
            frame_info = self.files[video_index][i]
            item = self._get_frame(frame_info)#item是dict
            clip.append(item)
        #for i in range(0,len(clip)):
        #    clip_depth=np.array(clip[i]['depth'])
        #    clip_depths.append(clip_depth)
        #print(np.shape(clip_depths))
        #clip_depths=np.array(clip_depths)      
        #clip_=torch.cat([clip_depths[0,:],clip_depths[1,:],clip_depths[2,:]],0)
        #clip_ = self.transforms(clip['depth'])
        
        #sample = self.transforms(sample)
        for clip_ in clip:
            sample = {'label': clip_['label'],'depth': clip_['depth'],'image':clip_['image']}
            sample = self.transforms(sample)
            clip_['label'] = sample['label']
            clip_['depth'] = sample['depth']
            clip_['image'] = sample['image']

        return clip

    def __len__(self):
        return len(self.clips)

def rgbd_datasets(name_list, split_list, config_path, root, training, transforms,
                    read_clip=True, random_reverse_clip=False, label_interval=1, frame_between_label_num=0, clip_len=4):

    if not isinstance(name_list, list):
        name_list = [name_list]
    if not isinstance(split_list, list):
        split_list = [split_list]
    if len(name_list) != len(split_list):
        raise ValueError("Dataset numbers must match split numbers")
    # read dataset config
    datasets_config = yaml.load(open(config_path))
    # get datasets
    dataset_list = []
    for name, split in zip(name_list, split_list):
        if name not in datasets_config.keys():
            raise ValueError("Error dataset name {}".format(name))

        dataset_config = datasets_config[name]
        dataset_config['name'] = name
        dataset_config['root'] = root
        dataset_config['split'] = split#將數據集分爲訓練數據集和測試數據集
        dataset_config['training'] = training
        dataset_config['transforms'] = transforms

        if "video_split" in dataset_config:
            dataset_config['label_interval'] = label_interval
            dataset_config['frame_between_label_num'] = frame_between_label_num
            if read_clip:
                dataset = VideoClipDataset(clip_len=clip_len,
                                        random_reverse_clip=random_reverse_clip,
                                        **dataset_config)

        dataset_list.append(dataset)

    if len(dataset_list) == 1:
        return dataset_list[0]
    else:
        return data.ConcatDataset(dataset_list)
