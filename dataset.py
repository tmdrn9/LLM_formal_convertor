#-*_ coding:utf-8 -*-
import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import ElectraModel, ElectraTokenizer
from transformers import AutoTokenizer, BartForConditionalGeneration

def RandomCrop(image, output_size):
    rows,cols = image.shape[:2]
    row_point = np.random.randint(0, rows - output_size)
    col_point = np.random.randint(0, cols - output_size)
    # dst = image[row_point[0]:row_point[0] + output_size, col_point[0]:col_point[0] + output_size]
    dst = image[row_point:row_point + output_size, col_point:col_point + output_size]
    return dst

# Train
def get_dataframe(k_fold, data_dir, data_folder, out_dim = 1):

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    data_folder = 'images/'
    df_train = pd.read_csv(os.path.join(data_dir, 'smilestyle_dataset.tsv'), delimiter='\t')

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    # df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, 'train', x))  # f'{x}.jpg'
    # df_train['org_img_name'] = df_train['image_name'].apply(lambda x: (x.split('_')[0]))

    # 원본데이터=0, 외부데이터=1
    # df_train['is_ext'] = 0
    df_train = df_train[['formal','informal']].dropna(axis=0)
    '''
    ####################################################
    교차 검증 구현 (k-fold cross-validation)
    ####################################################
    '''
    # 교차 검증을 위해 이미지 리스트들에 분리된 번호를 매김
    # img_ids = len(df_train['img_id'].unique())
    # print(f'Original dataset의 이미지수 : {img_ids}')
    df_test = df_train[2800:]
    df_train = df_train[:2800]
    return df_train, df_test

# Train
classes = {1:'a',2:'b'}
# Gray-Down-Crop(512)-까지 된 상태
class Dataset_train(Dataset): # train dataset 동적으로 만드는 class
    def __init__(self, csv, image_size=256, transform=None, mode='all'):
        self.csv = pd.concat([csv], ignore_index=True)
        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        ## model에 따라 tokenizer 변경
        # self.tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-formal-convertor")
        # self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")
        # self.tokenizer = AutoTokenizer.from_pretrained("heegyu/kobart-text-style-transfer")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        # self.tokenizer = AutoTokenizer.from_pretrained("google/electra-small-generator")

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        text_data = self.csv.iloc[index].informal
        target_data = self.csv.iloc[index].formal
        input_encoding = self.tokenizer(str(text_data), padding='max_length', return_tensors="pt", max_length=128, truncation=True)
        target_encoding = self.tokenizer(str(target_data), padding='max_length', return_tensors='pt', max_length=128, truncation=True)
        # input_encoding = self.tokenizer("반말로 바꿔주세요: " + text_data, return_tensors="pt")
        in_ids = input_encoding.input_ids
        in_attn = input_encoding.attention_mask

        return in_ids, in_attn, target_encoding, target_data
        #, torch.tensor(label).float() #, torch.tensor(factor).float()
#########################################

def get_transforms(image_size):
    transforms_train = albumentations.Compose([
        # albumentations.RandomBrightness(limit=0.1, p=0.75),
        # albumentations.RandomContrast(limit=0.1, p=0.75),
        # albumentations.CLAHE(clip_limit=2.0, p=0.3),
        # albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        # albumentations.Cutout(max_h_size=int(image_size * 0.1), max_w_size=int(image_size * 0.1), num_holes=1, p=0.7),
        # albumentations.RandomGamma(gamma_limit=(80, 120), eps=None,always_apply=False, p=0.5),
        albumentations.Normalize(mean=(0.5419184366861979),std=(0.14091745018959045))
        # shift
    ])

    transforms_val = albumentations.Compose([
        albumentations.Normalize(mean=(0.5419184366861979),std=(0.14091745018959045))
    ])

    return transforms_train, transforms_val

