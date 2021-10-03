import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import os
from src.utils import Logger, AverageMeter, calculate_metrics, calculate_gap
from src.models.model3 import Model
from src.data.dataset import TencentDataset2, create_val_transforms, create_train_transforms
import yaml

def eval_model(model, epoch, eval_loader, is_save=True):
    model.eval()
    losses = AverageMeter()
    gaps = AverageMeter()
    eval_process = tqdm(eval_loader)
    with torch.no_grad():
        for i, (video_feature,video_feature2, audio_feature, text, image, label) in enumerate(eval_process):
            if i > 0:
                eval_process.set_description( "Epoch: %d, Loss: %.4f, GAP: %.4f" %
                                              (epoch, losses.avg.item(), gaps.avg.item()))
            video_feature = Variable(video_feature.cuda())
            video_feature2 = Variable(video_feature2.cuda())

            audio_feature = Variable(audio_feature.cuda())
            text = Variable(text.cuda())
            image = Variable(image.cuda())
            label = Variable(label.float().cuda())
            y_pred = model(video_feature,video_feature2, audio_feature, text, image)
            y_pred = nn.Sigmoid().cuda()(y_pred)
            loss = criterion(y_pred, label)
            gap = calculate_gap(y_pred.data.cpu().numpy(), label.data.cpu().numpy())
            losses.update(loss.cpu(), image.size(0))
            gaps.update(gap, image.size(0))

    if is_save:
        train_logger.log(phase="val", values={
            'epoch': epoch,
            'loss': format(losses.avg.item(), '.4f'),
            'GAP': format(gaps.avg.item(), '.4f'),
            'lr': optimizer.param_groups[0]['lr']
        })
    print("Val:\t Loss:{0:.4f} \t GAP:{1:.4f}".format(losses.avg, gaps.avg))

    return gaps.avg


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
#                 print(name)
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


def train_model(model, criterion, optimizer, epoch):
    model.train()
    losses = AverageMeter()
    gaps = AverageMeter()
    training_process = tqdm(train_loader)
    # 初始化
#     pgd = PGD(model)
    fgm = FGM(model)

    for i, (video_feature,video_feature2, audio_feature, text, image, label) in enumerate(training_process):
        if i > 0:
            training_process.set_description("Epoch: %d, Loss: %.4f, GAP: %.4f" % (epoch, losses.avg.item(), gaps.avg.item()))
        video_feature = Variable(video_feature.cuda())
        video_feature2 = Variable(video_feature2.cuda())
        audio_feature = Variable(audio_feature.cuda())
        text = Variable(text.cuda())
        image = Variable(image.cuda())
        label = Variable(label.float().cuda())
        # Forward pass: Compute predicted y by passing x to the network
        y_pred = model(video_feature,video_feature2, audio_feature, text, image)
        y_pred = nn.Sigmoid().cuda()(y_pred)
        #image_output = nn.Sigmoid().cuda()(y_pred)
        loss = criterion(y_pred, label)
        gap = calculate_gap(y_pred.data.cpu().numpy(), label.data.cpu().numpy())
        losses.update(loss.cpu(), image.size(0))
        gaps.update(gap, image.size(0))
        optimizer.zero_grad()
        loss.backward() # 反向传播，得到正常的grad
        #         # 对抗训练
        fgm.attack() # 在embedding上添加对抗扰动
        y_pred_adv = model(video_feature, video_feature2, audio_feature, text, image)
        y_pred_adv = nn.Sigmoid().cuda()(y_pred_adv)
        loss_adv = criterion(y_pred_adv, label)
        loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore() # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()

    scheduler.step()
    train_logger.log(phase="train", values={
        'epoch': epoch,
        'loss': format(losses.avg.item(), '.4f'),
        'GAP': format(gaps.avg.item(), '.4f'),
        'lr': optimizer.param_groups[0]['lr']
    })
    print("Train:\t Loss:{0:.4f} \t GAP:{1:.4f}". format(losses.avg, gaps.avg))
    return 'tac{:.4f}'.format(gaps.avg.item(), '.4f')

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#setup_seed(2021)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/config.example.yaml',type=str)
    parser.add_argument("--k", default=-1, help="The value of K Fold", type=int)
    args = parser.parse_args()
    print(args.config)
    config = yaml.load(open(args.config))
    print(config)
    # 数据路劲
    root_path = 'tagging/tagging_dataset_train_5k'
    gt_txt = 'tagging/tagging_info.txt'
    class_txt = 'tagging/label_id.txt'
    vocab = 'tagging/vocab.txt'

    batch_size = config['batch_size']
    test_batch_size = 32
    input_size = 224
    text_dim = config['text_max_len']
    epoch_start = 1
    num_epochs = epoch_start + config['num_epoch']
    save_per_epoch = 1
    device_id = 0  # set the gpu id
    lr = 1e-3
    model_name = config['save_name']  # NextVLAD
    writeFile = 'output/logs/' + model_name #+ '_' + str(input_size)
    store_name = 'output/weights/' + model_name #+ '_' + str(input_size)
    k_fold = args.k
    if k_fold != -1:
        writeFile += f'_k{k_fold}'
        store_name += f'_k{k_fold}'

    if store_name and not os.path.exists(store_name):
        os.makedirs(store_name)

    model_path = None
    # Load network
    model = Model(config=config['ModelConfig'],hidden_dim=512,image_model_name='efficientnet-b0', num_classes=82)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print('Model found in {}'.format(model_path))
    else:
        print('No network found, initializing random network.')
    model = model.cuda()
    criterion = nn.BCELoss()
    is_train = True
    if is_train:
        train_logger = Logger(model_name=writeFile, header=['epoch', 'loss', 'GAP', 'lr'])
        params_list = []
        weight_decay = 4e-4
        for name,param in model.named_parameters():
            if param.requires_grad == False:
                continue
            if "video_model" in name or "audio_model" in name:
                temp_lr = lr * 0.1
            elif "text_bert" in name:
                temp_lr = lr * 0.01
            elif "image_model" in name:
                temp_lr = lr * 0.01
            elif "classifier" in name:
                temp_lr = lr * 10  #* 0.01# * 10
            elif "labelGCN" in name:
                temp_lr = lr * 0.01
            else:
                temp_lr = lr
            if 'bn' not in name or 'bias' not in name:
                params_list.append({'params':param,'weight_decay':weight_decay ,'lr': temp_lr})
            else:
                params_list.append({'params':param,'weight_decay':0.0 ,'lr': temp_lr})
        print('----', lr)
        optimizer = optim.AdamW(params_list, lr=lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8)

        xdl = TencentDataset2(root_path=root_path, gt_txt=gt_txt, class_txt=class_txt, vocab=vocab, phase='train',
                             transform=create_train_transforms(size=input_size), max_frame=config['max_frame'],text_dim=text_dim, k_fold=k_fold,model_name=model_name)
        train_loader = DataLoader(xdl, batch_size=batch_size, shuffle=True, num_workers=16,drop_last=True)
        train_dataset_len = len(xdl)
        xdl_eval = TencentDataset2(root_path=root_path, gt_txt=gt_txt, class_txt=class_txt, vocab=vocab, phase='val',
                                  transform=create_val_transforms(size=input_size),max_frame=config['max_frame'], text_dim=text_dim, k_fold=k_fold,model_name=model_name)
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('train_dataset_len:', train_dataset_len, 'eval_dataset_len:', eval_dataset_len)

        best_acc = 0.5 if epoch_start == 1 else eval_model(model, epoch_start-1, eval_loader, is_save=False)
        for epoch in range(epoch_start, num_epochs):
            tac = train_model(model,criterion, optimizer, epoch)
            if epoch % save_per_epoch == 0 or epoch == num_epochs - 1:
                acc = eval_model(model, epoch, eval_loader)
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), '{}/{}_{}_acc{:.4f}.pth'.format(store_name, epoch, tac, acc))
            print('current best acc:', best_acc)









