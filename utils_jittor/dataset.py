# -*- encoding: utf-8 -*-
# !/usr/bin/env python

import os
import jittor
import jittor.dataset as dataset
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

### AVA
params_ava = {
    'dataset_name':'AVA',
    'buffer_root':'./preprocess/',
    'preload_file':'preload_AVA.txt',
    'img_path':'images/',
    'img_list':'AVA.txt',
    'split':'aesthetics_image_lists/generic_test.jpgl',
    'corrupted_ids': [2113, 2878, 770406, 729377, 2761, 3101, 3937],
}

### AADB
params_aadb = {
    'dataset_name':'AADB',
    'buffer_root': './preprocess/',
    'preload_file': 'preload_AADB.txt',
    'img_path': 'datasetImages_originalSize/', 
    'meta_data': 'AADB_AllinAll.csv',
    'img_list': 'AADB_imgListFiles_label/imgListFiles_label'
}

ImageFile.LOAD_TRUNCATED_IMAGES = True
class Dataset(dataset.Dataset):
    def __init__(self, data, transform=None):
        '''
            Initialization:
        '''
        super().__init__()
        self.drop_last = True
        self.paths, self.rates, self.cls, self.ratios = data
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        assert len(self.paths) == len(self.rates)
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img_rate = self.rates[index]
        img_rate = img_rate / np.sum(img_rate)
        img_cls = self.cls[index]
        img_ratio = self.ratios[index]
        im_ = Image.open(img_path) 
        # img_ratio = float(im_.size[0]/im_.size[1])
        X1 = jittor.array(self.transform(im_.convert('RGB')))
        img_ratio =  jittor.float(img_ratio)
        label = jittor.float(img_rate)
        cls_label = jittor.int64(img_cls)
        return X1, img_ratio, label, cls_label

class DatasetCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.X1 = jittor.stack(transposed_data[0], 0)
        self.ratio = jittor.stack(transposed_data[1], 0)
        self.label = jittor.stack(transposed_data[2], 0)
        self.cls_label = jittor.stack(transposed_data[3], 0)
        
    def pin_memory(self):
        # TODO: no pin_memory in jittor
        #self.X1 = self.X1.pin_memory()
        #self.ratio = self.ratio.pin_memory()
        #self.label = self.label.pin_memory()
        #self.cls_label = self.cls_label.pin_memory()
        return self

def collate_wrapper(batch):
    return DatasetCustomBatch(batch)

def add_aes_item(img_score_lists, datadict, params_aadb_dataset, dir_root):
    img_paths, img_rates, img_ratios, img_cls= [], [], [], []
    cnt_zero = 0
    for i in img_score_lists:
        i = i.strip('\n').split(' ')
        _path = os.path.join(dir_root, params_aadb_dataset['img_path'], i[0])
        _cls = 1 if float(i[1]) > 0.5 else 0
        try:
            _im = Image.open(_path)
            _ratio = float(_im.size[0]/_im.size[1])
            _rate = [int(datadict[i[0]][k]) for k in range(1, 6)]
            if (np.sum(_rate)<1):
                cnt_zero = cnt_zero+1
                continue
            img_paths.append(_path)
            img_rates.append(_rate)
            img_cls.append(_cls)
            img_ratios.append(_ratio) 
        except Exception as e:
            print(e, _path)
    return img_paths, img_rates, img_ratios, img_cls

def loadinfo_AADB(dir_root, params_dataset=params_aadb):
    buffer_root = params_dataset['buffer_root']
    imgfile_path = os.path.join(dir_root, params_dataset['img_path'])
    split_path = os.path.join(dir_root, params_dataset['img_list'])
    meta_path = os.path.join(dir_root, params_dataset['meta_data'])
    datadict = {}     
    data = pd.read_csv(meta_path)
    for j in range(data['Answer.overallScore1'].shape[0]):
        for i in range(10):
            str1 = data['Input.image_url'+str(i+1)][j].split('.')[0].split('/')[-1]
            str2 = data['Input.image_url'+str(i+1)][j].split('/')[3]
            str3 = data['Input.image_url'+str(i+1)][j].split('/')[-1]
            img_name = str1 + '_' + str2 + '_'+str3
            img_score = int(data['Answer.overallScore'+str(i+1)][j])
            try:
                datadict[img_name][img_score] = datadict[img_name][img_score]+1       
            except Exception as e:
                datadict[img_name]=[0]*6 # score 0-5
                if os.path.isfile(os.path.join(imgfile_path, img_name)):
                    datadict[img_name][img_score] = datadict[img_name][img_score]+1       
  
    with open(os.path.join(split_path, 'imgListTrainRegression_score.txt'),'r') as ff:
        lines = ff.readlines()
        img_paths_train, img_rates_train, img_ratios_train, img_cls_train = \
        add_aes_item(lines, datadict, params_dataset, dir_root)
    with open(os.path.join(split_path, 'imgListValidationRegression_score.txt'),'r') as ff:
        lines = ff.readlines()
        img_paths_val, img_rates_val, img_ratios_val, img_cls_val = \
        add_aes_item(lines, datadict, params_dataset, dir_root)
    with open(os.path.join(split_path, 'imgListTestRegression_score.txt'),'r') as ff:
        lines = ff.readlines()
        img_paths_test, img_rates_test, img_ratios_test, img_cls_test = \
        add_aes_item(lines, datadict, params_dataset, dir_root)

    imginfo = [img_paths_train, img_rates_train, img_cls_train, img_ratios_train, \
               img_paths_test, img_rates_test, img_cls_test, img_ratios_test, \
               img_paths_val, img_rates_val, img_cls_val, img_ratios_val]
    print(" "*5 + f'AADB dataset info preloaded in {buffer_root}: #{len(img_paths_train)} train #{len(img_paths_val)} val #{len(img_paths_test)} test')
    return imginfo
          
def loadinfo_AVA(dir_root, params_dataset=params_ava, standard=None):
    buffer_root = params_dataset['buffer_root']
    img_list = os.path.join(dir_root, params_dataset['img_list'])
    img_path = os.path.join(dir_root, params_dataset['img_path'])
    std_split = os.path.join(dir_root, params_dataset['split'])
    img_paths, img_rates, img_ratios, img_cls = [], [], [], [] 
    img_paths_test, img_rates_test, img_ratios_test, img_cls_test = [], [], [], []
    te_id = []
    
    if standard is not None:
        with open(std_split, mode='r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                te_id.append(int(line.split()[0]))
    else:
        with open(os.path.join(dir_root, 'val.txt'),'r') as f:
            lines = f.readlines()
            for i in lines:
                i =  i.strip('\n').split(' ')
                te_id.append(int(i[0].split('.')[0]))
            
    with open(img_list, mode='r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            token = line.split()
            _img_id =  int(token[1])
            if _img_id in set(params_dataset['corrupted_ids']):
                pass
            _path = os.path.join(img_path, str(_img_id)+'.jpg')
            _rate = [int(token[i]) for i in range(2, 12)]
            _cls = 1 if avg_score(_rate) > 5 else 0
            try:
                _im = Image.open(_path)
                _ratio = float(_im.size[0]/_im.size[1])
                if _img_id in te_id:
                    img_paths_test.append(_path)
                    img_rates_test.append(_rate)
                    img_cls_test.append(_cls)
                    img_ratios_test.append(_ratio)
                else:
                    img_paths.append(_path)
                    img_rates.append(_rate)
                    img_cls.append(_cls)
                    img_ratios.append(_ratio)
            except Exception as e:
                print("error", e, _path)
                
    print(" "*5 + f'AVA dataset info preloaded in {buffer_root}!: #{len(img_paths)} trainval #{len(img_paths_test)} test')
    imginfo = [img_paths, img_rates, img_cls, img_ratios, img_paths_test, img_rates_test, img_cls_test, img_ratios_test]
    return imginfo
          
def preload_img(params_dataset, dir_root):

    buffer_root = params_dataset['buffer_root']
    filename = params_dataset['preload_file']
    if os.path.exists(os.path.join(buffer_root, filename)):
        b = open(os.path.join(buffer_root, filename), "r")
        out = b.read()
        imginfo = json.loads(out)
    else:
        print("Preprocessing dataset...")
        if params_dataset['dataset_name']=='AADB':
              imginfo = loadinfo_AADB(dir_root)
        elif params_dataset['dataset_name']=='AVA':
              imginfo = loadinfo_AVA(dir_root)
        c_list = json.dumps(imginfo)
        if jittor.rank == 0:
            a = open(os.path.join(buffer_root, filename),"w")
            a.write(c_list)
            a.close()
            print("Preloading file saved!")
    return imginfo

def avg_score(rate):
    label_scale = [float(i+1) for i in range(10)]
    r = rate/np.sum(rate)
    score = np.sum(label_scale * r)
    return score
