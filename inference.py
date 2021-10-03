import warnings
import sys

warnings.filterwarnings("ignore")
sys.path.append('..')
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import json
from torch.utils.data import DataLoader
import numpy as np
import os
from src.models.model import Model
from src.data.dataset import create_val_transforms, TencentDatasetInference
import pickle

np.set_printoptions(suppress=True)


def inference(model, index_to_tag, eval_loader, top_k=20, output_label=False, output_file_name=""):
    total_index = 1
    output_result = {}
    if output_label:
        pseudo_label_dict = {}
        pseudo_label_file = open(output_file_name, 'wb')
    with torch.no_grad():
        count = 0
        for i, (video_feature, audio_feature, text, image, video_names) in enumerate(eval_loader):
            video_feature = Variable(video_feature.cuda(device_id))
            audio_feature = Variable(audio_feature.cuda(device_id))
            text = Variable(text.cuda(device_id))
            image = Variable(image.cuda(device_id))
            if not is_ensemble:
                y_pred = model(video_feature, audio_feature, text, image)
                y_pred = nn.Sigmoid()(y_pred).data.cpu().numpy()
            else:
                y_pred = model[0](video_feature, audio_feature, text, image)
                y_pred = nn.Sigmoid()(y_pred).data.cpu().numpy()
                for model_i in model[1:]:
                    y_pred_i = model_i(video_feature, audio_feature, text, image)
                    y_pred_i = nn.Sigmoid()(y_pred_i).data.cpu().numpy()
                    y_pred += y_pred_i
                y_pred /= len(model)
                if output_label:
                    y_pred_list = []
                    for model_i in model:
                        y_pred_i = model_i(video_feature, audio_feature, text, image)
                        y_pred_i = nn.Sigmoid()(y_pred_i).data.cpu().numpy()
                        y_pred_list.append(y_pred_i)
            for index, video_name in enumerate(video_names):
                if output_label:
                    pred = y_pred[index]
                    pseudo_label_dict[video_name[:-4]] = pred
                pred = y_pred[index]
                pred_index = np.argsort(-pred)  # 逆序排序
                labels = [index_to_tag[idx] for idx in pred_index[:top_k]]  # 选择top k标签
                scores = pred[pred_index[:top_k]]  # 选择top k标签对应的得分
                cur_output = {}
                output_result[video_name] = cur_output
                cur_output["result"] = [
                    {"labels": labels, "scores": ["%.2f" % scores[i] for i in range(top_k)]}]
                if total_index % 100 == 0:
                    print("%d/%d" % (total_index, 5000))
                total_index += 1
    if output_label:
        pickle.dump(pseudo_label_dict, pseudo_label_file)
        pseudo_label_file.close()
    return output_result


def inference_from_all_file(files, index_to_tag, eval_loader, top_k=20):
    total_index = 1
    output_result = {}
    logits = []
    for file in files:
        label_file = open(file, 'rb')
        test_logit = pickle.load(label_file)
        label_file.close()
        logits.append(test_logit)
    with torch.no_grad():
        count = 0
        for i, (video_feature, audio_feature, text, image, video_names) in enumerate(eval_loader):
            for index, video_name in enumerate(video_names):
                video_name = video_name[:-4]
                y_pred = logits[0][video_name]
                for logit in logits[1:]:
                    y_pred += logit[video_name]
                y_pred /= len(logits)
                pred = y_pred
                pred_index = np.argsort(-pred)  # 逆序排序
                labels = [index_to_tag[idx] for idx in pred_index[:top_k]]  # 选择top k标签
                scores = pred[pred_index[:top_k]]  # 选择top k标签对应的得分
                cur_output = {}
                output_result[video_name + ".mp4"] = cur_output
                cur_output["result"] = [
                    {"labels": labels, "scores": ["%.2f" % scores[i] for i in range(top_k)]}]
                if total_index % 100 == 0:
                    print("%d/%d" % (total_index, 5000))
                total_index += 1
    return output_result


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(2021)

if __name__ == '__main__':
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.example.yaml', type=str)
    parser.add_argument('--model_name', default="", type=str)
    parser.add_argument('--output_pkl', default=1, type=int)
    parser.add_argument('--output_predict', default=0, type=int)
    parser.add_argument('--k', default=0, type=int)
    args = parser.parse_args()
    print(args.config)
    config = yaml.load(open(args.config))
    print(config)
    # 测试集
    root_path = 'tagging/tagging_dataset_test_5k_2nd'
    class_txt = 'tagging/label_id.txt'
    vocab = 'tagging/vocab.txt'

    test_batch_size = 32
    input_size = 224
    text_dim = config['text_max_len']
    device_id = 0  # set the gpu id
    model_name = config['save_name']

    if args.output_pkl == 1:
        xdl_eval = TencentDatasetInference(root_path=root_path, class_txt=class_txt, vocab=vocab,
                                           transform=create_val_transforms(size=input_size), text_dim=text_dim, model_name = model_name)
        eval_loader = DataLoader(xdl_eval, batch_size=test_batch_size, shuffle=False, num_workers=4)
        eval_dataset_len = len(xdl_eval)
        print('eval_dataset_len:', eval_dataset_len)
        for i in range(args.k):
            output_file_name = "output/pkl/{}_k{}.pkl".format(args.model_name, i)
            model_path = 'output/weights/{}_k{}'.format(args.model_name, i)
            files = os.listdir(model_path)
            files_score = [float(x.split("acc")[-1][:-4]) for x in files if ".pth" in x]
            model_path = 'output/weights/{}_k{}/{}'.format(args.model_name, i, files[np.argmax(files_score)])
            model = Model(config=config['ModelConfig'], hidden_dim=512, image_model_name='efficientnet-b0',
                          num_classes=82)
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
            print('Model found in {}'.format(model_path))
            model = model.cuda(device_id)
            model.eval()

            json_data = inference(model, xdl_eval.index_to_tag, eval_loader, output_label=output_label,
                                  output_file_name=output_file_name)
            with open(json_path, 'w', encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)

    if args.output_predict == 1:
        store_name = 'output/results'  # + str(input_size)
        if store_name and not os.path.exists(store_name):
            os.makedirs(store_name)
        json_path = os.path.join(store_name, 'result.json')

        predict_root = 'output/pkl'
        files = os.listdir(predict_root)
        print(files)
        files = [os.path.join(predict_root, x) for x in files if ".pkl" in x]
        print(files)
        json_data = inference_from_all_file(files, xdl_eval.index_to_tag, eval_loader)
        with open(json_path, 'w', encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
