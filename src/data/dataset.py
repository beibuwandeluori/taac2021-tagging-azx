from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import time
from .preprocess import extract_dict, TextPreprocess

from albumentations.pytorch import ToTensorV2
from albumentations import Compose, RandomBrightnessContrast, RandomCrop, CenterCrop,\
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise,\
    GaussianBlur, Resize, Normalize, RandomRotate90, Cutout, GridDropout, CoarseDropout, MedianBlur
import torch
from sklearn.model_selection import KFold


# ———————————————————————————————#
def create_train_transforms(size=224):
    return Compose([
        HorizontalFlip(),
        GaussNoise(p=0.1),
        GaussianBlur(p=0.1),        
        Resize(height=size, width=size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )


def create_val_transforms(size=224):
    return Compose([
        Resize(height=size, width=size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
# ———————————————————————————————#

# 同步对应打乱两个数组
def shuffle_two_array(a, b, seed=None):
    state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(a)
    np.random.set_state(state)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(b)
    return a, b

# 读取视频id和标签
def read_gt_txt(txt_path, tag_to_index):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    video_ids = []
    labels = []
    for line in lines:
        line = line.split('	')
        video_id = line[0][:-4]
        video_ids.append(video_id)
        label = np.zeros(shape=(82, ), dtype=np.float32)
        label_names = line[1].strip().split(',')
        for name in label_names:
            try:
                class_index = tag_to_index[name]
            except:
                print(line)
            label[class_index] = 1
        # print(label.dtype, type(label))
        labels.append(label)

    return video_ids, labels

# 分割训练集和验证集
def split_data(video_ids, labels, phase='train', val_size=0.1, seed=2021):
    video_ids, labels = shuffle_two_array(video_ids, labels, seed=seed)
    lenght = len(video_ids)
    if phase == 'train':
        video_ids = video_ids[:int(lenght * (1-val_size))]
        labels = labels[:int(lenght * (1 - val_size))]
    else:
        video_ids = video_ids[int(lenght * (1 - val_size)):]
        labels = labels[int(lenght * (1 - val_size)):]
    return video_ids, labels

# 分割训练集和验证集
def split_data_by_k_fold(video_ids, labels, phase='train', n_splits=5, k=1):
    kf = KFold(n_splits=n_splits)
    for i, (train, valid) in enumerate(kf.split(X=video_ids, y=labels)):
        if i == k:
            train_indexs, valid_indexs = train, valid
    if phase == 'train':
        x = np.array(video_ids)[train_indexs]
        y = np.array(labels)[train_indexs]
    else:
        x = np.array(video_ids)[valid_indexs]
        y = np.array(labels)[valid_indexs]

    return x, y

# 读取训练数据
class TencentDataset(Dataset):
    def __init__(self, root_path='/pubdata/chenby/Tencent/VideoStructuring/dataset/tagging/tagging_dataset_train_5k',
                 gt_txt='/pubdata/chenby/Tencent/VideoStructuring/dataset/tagging/GroundTruth/tagging_info.txt',
                 class_txt='/pubdata/chenby/Tencent/VideoStructuring/dataset/label_id.txt',
                 vocab='/pubdata/chenby/Tencent/VideoStructuring/MultiModal-Tagging/pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt',
                 phase='train',with_occur_feat=False, transform=None, max_frame=120, text_dim=128,k_fold=-1,model_name=""):
        self.root_path = root_path
        self.gt_txt = gt_txt
        self.class_txt = class_txt
        self.vocab = vocab
        self.phase = phase
        self.transform = transform
        self.max_frame = max_frame  # 视频的最大帧数
        self.text_dim = text_dim  # 使用的字符长度
        self.text_preprocess = TextPreprocess(max_len=text_dim, vocab=vocab)
        self.index_to_tag, self.tag_to_index = extract_dict(self.class_txt)
        self.video_ids, self.labels = read_gt_txt(self.gt_txt, self.tag_to_index)
        if k_fold == -1:
            self.video_ids, self.labels = split_data(self.video_ids, self.labels, phase=phase, val_size=0.1, seed=2021)
        else:
            self.video_ids, self.labels = split_data_by_k_fold(self.video_ids, self.labels, phase=phase, n_splits=10, k=k_fold)
        self.phase = phase
        self.video_root = os.path.join(self.root_path, 'video_npy/Youtube8M/tagging')
        self.video_root_H14 = os.path.join(self.root_path, 'video_npy/VIT_H14/tagging')  # 1280 dimension
        self.video_root_L16 = os.path.join(self.root_path, 'video_npy/VIT_L16/tagging')  # 1024 dimension
        self.video_root_B16 = os.path.join(self.root_path, 'video_npy/VIT_B16/tagging')  # 768 dimension
        self.audio_root = os.path.join(self.root_path, 'audio_npy/Vggish/tagging')
        self.text_root = os.path.join(self.root_path, 'text_txt/tagging')
        self.image_root = os.path.join(self.root_path, 'image_jpg/tagging')
        
        self.with_occur_feat = with_occur_feat
        self.model_name = model_name
        
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        # 加载视频特征 shape=(max_frame, 1024)
        if self.model_name == "model1": 
            video_feature = np.zeros(shape=(self.max_frame, 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_t = video_feature_h14
                video_shape = video_feature_t.shape
                video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
    
        if self.model_name == "model2":
            video_feature = np.zeros(shape=(self.max_frame, 1024+1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_l16 = np.load(os.path.join(self.video_root_L16, f'{video_id}.npy'))
                video_feature_t = np.concatenate((video_feature_h14,video_feature_l16), axis=1)
                video_shape = video_feature_t.shape
                video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
                
        audio_feature = np.zeros(shape=(self.max_frame, 128), dtype=np.float32)
        try:
            audio_feature_t = np.load(os.path.join(self.audio_root, f'{video_id}.npy'))
            audio_shape = audio_feature_t.shape
            audio_feature[:audio_feature_t.shape[0], :] = audio_feature_t
        except:
            print(os.path.join(self.audio_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
        # 读取文字 shape=(128,)
        text_path = os.path.join(self.text_root, f'{video_id}.txt')
        text = self.text_preprocess(text_path)
        # 读取图片
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_root, f'{video_id}.jpg')), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]         
        label = self.labels[index]
        
        if self.with_occur_feat:
            occur = np.load(os.path.join(self.occur_root,f'{video_id}.npy'))
            occur = occur.reshape(-1)
            return video_feature, audio_feature, text, image,occur, label
        else:
            return video_feature, audio_feature, text, image, label

# 读取测试数据
class TencentDatasetInference(Dataset):
    def __init__(self, root_path='/pubdata/chenby/Tencent/VideoStructuring/dataset/tagging/tagging_dataset_test_5k',
                 class_txt='/pubdata/chenby/Tencent/VideoStructuring/dataset/label_id.txt',
                 vocab='/pubdata/chenby/Tencent/VideoStructuring/MultiModal-Tagging/pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt',
                 transform=None, max_frame=120, text_dim=128, model_name = 'model1'):
        self.root_path = root_path
        self.class_txt = class_txt
        self.vocab = vocab
        self.transform = transform
        self.max_frame = max_frame  # 视频的最大帧数
        self.text_dim = text_dim  # 使用的字符长度
        self.text_preprocess = TextPreprocess(max_len=text_dim, vocab=vocab)
        self.index_to_tag, self.tag_to_index = extract_dict(self.class_txt)
        # self.text_preprocess = TextPreprocess(max_len=text_dim)

        self.video_root = os.path.join(self.root_path, 'video_npy/Youtube8M/tagging')
        self.video_root_H14 = os.path.join(self.root_path, 'video_npy/VIT_H14/tagging') # 1280
        self.video_root_L16 = os.path.join(self.root_path, 'video_npy/VIT_L16/tagging') # 1024
        self.video_root_B16 = os.path.join(self.root_path, 'video_npy/VIT_B16/tagging') # 768
        self.audio_root = os.path.join(self.root_path, 'audio_npy/Vggish/tagging')
        self.text_root = os.path.join(self.root_path, 'text_txt/tagging')
        self.image_root = os.path.join(self.root_path, 'image_jpg/tagging')

        self.video_ids = sorted([video_name[:-4] for video_name in os.listdir(self.video_root)])
        print(len(self.video_ids), self.video_ids[:2])

        self.model_name = model_name

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        # 加载视频特征 shape=(max_frame, 1024)
#         video_feature = np.zeros(shape=(self.max_frame, 1024), dtype=np.float32)
        if self.model_name == "model1":
            video_feature = np.zeros(shape=(self.max_frame, 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_t = video_feature_h14
                video_shape = video_feature_t.shape
                video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')

        if self.model_name == "model2":
            video_feature = np.zeros(shape=(self.max_frame, 1024 + 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_l16 = np.load(os.path.join(self.video_root_L16, f'{video_id}.npy'))
                video_feature_t = np.concatenate((video_feature_h14, video_feature_l16), axis=1)
                video_shape = video_feature_t.shape
                video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')

        # 加载语音特征 shape=(max_frame, 128)
        audio_feature = np.zeros(shape=(self.max_frame, 128), dtype=np.float32)
        try:
            audio_feature_t = np.load(os.path.join(self.audio_root, f'{video_id}.npy'))
            audio_feature[:audio_feature_t.shape[0], :] = audio_feature_t
        except:
            print(os.path.join(self.audio_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
        # 读取文字 shape=(128,)
        text_path = os.path.join(self.text_root, f'{video_id}.txt')
        text = self.text_preprocess(text_path)
        # 读取图片
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_root, f'{video_id}.jpg')), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        video_name = video_id + '.mp4'

        return video_feature, audio_feature, text, image, video_name


class TencentDataset2(Dataset):
    def __init__(self, root_path='/pubdata/chenby/Tencent/VideoStructuring/dataset/tagging/tagging_dataset_train_5k',
                 gt_txt='/pubdata/chenby/Tencent/VideoStructuring/dataset/tagging/GroundTruth/tagging_info.txt',
                 class_txt='/pubdata/chenby/Tencent/VideoStructuring/dataset/label_id.txt',
                 vocab='/pubdata/chenby/Tencent/VideoStructuring/MultiModal-Tagging/pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt',
                 phase='train', with_occur_feat=False, transform=None, max_frame=120, text_dim=128, k_fold=-1,
                 model_name=""):
        self.root_path = root_path
        self.gt_txt = gt_txt
        self.class_txt = class_txt
        self.vocab = vocab
        self.phase = phase
        self.transform = transform
        self.max_frame = max_frame  # 视频的最大帧数
        self.text_dim = text_dim  # 使用的字符长度
        self.text_preprocess = TextPreprocess(max_len=text_dim, vocab=vocab)
        self.index_to_tag, self.tag_to_index = extract_dict(self.class_txt)
        self.video_ids, self.labels = read_gt_txt(self.gt_txt, self.tag_to_index)
        if k_fold == -1:
            self.video_ids, self.labels = split_data(self.video_ids, self.labels, phase=phase, val_size=0.1, seed=2021)
        else:
            self.video_ids, self.labels = split_data_by_k_fold(self.video_ids, self.labels, phase=phase, n_splits=10,
                                                               k=k_fold)
        self.phase = phase
        self.video_root = os.path.join(self.root_path, 'video_npy/Youtube8M/tagging')
        self.video_root_H14 = os.path.join(self.root_path, 'video_npy/VIT_H14/tagging')  # 1280 dimension
        self.video_root_L16 = os.path.join(self.root_path, 'video_npy/VIT_L16/tagging')  # 1024 dimension
        self.video_root_B16 = os.path.join(self.root_path, 'video_npy/VIT_B16/tagging')  # 768 dimension
        self.audio_root = os.path.join(self.root_path, 'audio_npy/Vggish/tagging')
        self.text_root = os.path.join(self.root_path, 'text_txt/tagging')
        self.image_root = os.path.join(self.root_path, 'image_jpg/tagging')

        self.with_occur_feat = with_occur_feat
        self.model_name = model_name

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        # 加载视频特征 shape=(max_frame, 1024)
        if self.model_name == "model1":
            video_feature = np.zeros(shape=(self.max_frame, 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_t = video_feature_h14
                video_shape = video_feature_t.shape
                video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')

        if self.model_name == "model3":
            # video_feature = np.zeros(shape=(self.max_frame, 1024 + 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_l16 = np.load(os.path.join(self.video_root_L16, f'{video_id}.npy'))
                video_feature = np.zeros(shape=(self.max_frame, 1280), dtype=np.float32)
                video_feature_t = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature[:video_feature_t.shape[0], :] = video_feature_t

                video_feature2 = np.zeros(shape=(self.max_frame, 1024), dtype=np.float32)
                video_feature_t2 = np.load(os.path.join(self.video_root_L16, f'{video_id}.npy'))
                video_feature2[:video_feature_t.shape[0], :] = video_feature_t2
                # video_feature_t = np.concatenate((video_feature_h14, video_feature_l16), axis=1)
                # video_shape = video_feature_t.shape
                # video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')

        audio_feature = np.zeros(shape=(self.max_frame, 128), dtype=np.float32)
        try:
            audio_feature_t = np.load(os.path.join(self.audio_root, f'{video_id}.npy'))
            audio_shape = audio_feature_t.shape
            audio_feature[:audio_feature_t.shape[0], :] = audio_feature_t
        except:
            print(os.path.join(self.audio_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
        # 读取文字 shape=(128,)
        text_path = os.path.join(self.text_root, f'{video_id}.txt')
        text = self.text_preprocess(text_path)
        # 读取图片
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_root, f'{video_id}.jpg')), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = self.labels[index]

        if self.with_occur_feat:
            occur = np.load(os.path.join(self.occur_root, f'{video_id}.npy'))
            occur = occur.reshape(-1)
            return video_feature, audio_feature, text, image, occur, label
        else:
            return video_feature, video_feature2, audio_feature, text, image, label


# 读取测试数据
class TencentDatasetInference2(Dataset):
    def __init__(self, root_path='/pubdata/chenby/Tencent/VideoStructuring/dataset/tagging/tagging_dataset_test_5k',
                 class_txt='/pubdata/chenby/Tencent/VideoStructuring/dataset/label_id.txt',
                 vocab='/pubdata/chenby/Tencent/VideoStructuring/MultiModal-Tagging/pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt',
                 transform=None, max_frame=120, text_dim=128, model_name = 'model3'):
        self.root_path = root_path
        self.class_txt = class_txt
        self.vocab = vocab
        self.transform = transform
        self.max_frame = max_frame  # 视频的最大帧数
        self.text_dim = text_dim  # 使用的字符长度
        self.text_preprocess = TextPreprocess(max_len=text_dim, vocab=vocab)
        self.index_to_tag, self.tag_to_index = extract_dict(self.class_txt)
        # self.text_preprocess = TextPreprocess(max_len=text_dim)

        self.video_root = os.path.join(self.root_path, 'video_npy/Youtube8M/tagging')
        self.video_root_H14 = os.path.join(self.root_path, 'video_npy/VIT_H14/tagging')  # 1280
        self.video_root_L16 = os.path.join(self.root_path, 'video_npy/VIT_L16/tagging')  # 1024
        self.video_root_B16 = os.path.join(self.root_path, 'video_npy/VIT_B16/tagging')  # 768
        self.audio_root = os.path.join(self.root_path, 'audio_npy/Vggish/tagging')
        self.text_root = os.path.join(self.root_path, 'text_txt/tagging')
        self.image_root = os.path.join(self.root_path, 'image_jpg/tagging')

        self.video_ids = sorted([video_name[:-4] for video_name in os.listdir(self.video_root)])
        print(len(self.video_ids), self.video_ids[:2])
        self.model_name = model_name

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        # 加载视频特征 shape=(max_frame, 1024)
        #         video_feature = np.zeros(shape=(self.max_frame, 1024), dtype=np.float32)
        if self.model_name == "model1":
            video_feature = np.zeros(shape=(self.max_frame, 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_t = video_feature_h14
                video_shape = video_feature_t.shape
                video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')


        if self.model_name == "model3":
            # video_feature = np.zeros(shape=(self.max_frame, 1024 + 1280), dtype=np.float32)
            try:
                video_feature_h14 = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature_l16 = np.load(os.path.join(self.video_root_L16, f'{video_id}.npy'))
                video_feature = np.zeros(shape=(self.max_frame, 1280), dtype=np.float32)
                video_feature_t = np.load(os.path.join(self.video_root_H14, f'{video_id}.npy'))
                video_feature[:video_feature_t.shape[0], :] = video_feature_t

                video_feature2 = np.zeros(shape=(self.max_frame, 1024), dtype=np.float32)
                video_feature_t2 = np.load(os.path.join(self.video_root_L16, f'{video_id}.npy'))
                video_feature2[:video_feature_t.shape[0], :] = video_feature_t2
                # video_feature_t = np.concatenate((video_feature_h14, video_feature_l16), axis=1)
                # video_shape = video_feature_t.shape
                # video_feature[:video_feature_t.shape[0], :] = video_feature_t
            except:
                print(os.path.join(self.video_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
        # 加载语音特征 shape=(max_frame, 128)
        audio_feature = np.zeros(shape=(self.max_frame, 128), dtype=np.float32)
        try:
            audio_feature_t = np.load(os.path.join(self.audio_root, f'{video_id}.npy'))
            audio_feature[:audio_feature_t.shape[0], :] = audio_feature_t
        except:
            print(os.path.join(self.audio_root, f'{video_id}.npy'), 'is not exist!!!!!!!!!!!!!!!')
        # 读取文字 shape=(128,)
        text_path = os.path.join(self.text_root, f'{video_id}.txt')
        text = self.text_preprocess(text_path)
        # 读取图片
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_root, f'{video_id}.jpg')), cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        video_name = video_id + '.mp4'

        return video_feature, video_feature2, audio_feature, text, image, video_name

