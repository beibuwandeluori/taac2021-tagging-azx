from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision
import os
import torch
from transformers import BertTokenizer, BertModel

def get_efficientnet(model_name='efficientnet-b0', num_classes=2):
    net = EfficientNet.from_pretrained(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=82):
        super().__init__()
        self.dense1 = nn.Linear(out_size, 1024)
        self.norm_0 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.7)
        self.out_proj = nn.Linear(1024, num_classes)

    def forward(self, features):
        x = torch.cat(features, -1)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.relu(self.norm_0(x))
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x

class NextVLAD(nn.Module):
    def __init__(self, embed_dim=128,output_dim=1024, nextvlad_cluster_size=64, groups=16, expansion=2):
        '''
        :param embed_dim: audio 128   video 1024
        :param nextvlad_cluster_size: follow  config.tagging.5k.yaml
        :param groups: follow  config.tagging.5k.yaml
        :param expansion: follow  config.tagging.5k.yaml
        '''
        super(NextVLAD, self).__init__()
        self.embed_dim = embed_dim
        self.nextvlad_cluster_size = nextvlad_cluster_size
        self.groups = groups
        self.expansion = expansion
        self.linear1 = nn.Linear(self.embed_dim, self.expansion * self.embed_dim,bias=True)
        self.linear2 = nn.Linear(self.expansion * self.embed_dim, self.groups,bias=True)
        self.cluster_weights = nn.Parameter(torch.Tensor(self.expansion * self.embed_dim,
                                                         self.groups*self.nextvlad_cluster_size))
        feature_size = self.expansion * self.embed_dim // self.groups
        self.cluster2_weights = nn.Parameter(torch.Tensor(1,feature_size,self.nextvlad_cluster_size))
        self.activation_bn = nn.BatchNorm1d(self.groups*self.nextvlad_cluster_size)
        self.vlad_bn = nn.BatchNorm1d(self.nextvlad_cluster_size * feature_size)
        # 20210521          
        torch.nn.init.kaiming_normal(self.cluster_weights)
        torch.nn.init.kaiming_normal(self.cluster2_weights)

        
    def forward(self, input):
        '''
        :param input: [B,num_frames,embed_dim]
        :return:
        '''
        res = input
        mask = (input != 0.)[:,:,0]
        max_frames = input.shape[1]
        input = self.linear1(input) #  [B,num_frames,expansion* embed_dim]
        attention = self.linear2(input)
        # TODO attention加sigmoid
        attention = torch.sigmoid(attention)
        if mask is not None:
            attention = attention * mask.unsqueeze(-1)
        attention = attention.view([-1,max_frames*self.groups,1])
#         print(attention)
        feature_size = self.expansion * self.embed_dim // self.groups

        reshaped_input = input.view([-1, self.expansion * self.embed_dim])
        activation = reshaped_input.matmul(self.cluster_weights)
        activation = self.activation_bn(activation)
        activation = activation.view([-1, max_frames * self.groups, self.nextvlad_cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)  # B*1*embed_dim
        a = a_sum * self.cluster2_weights
        activation = activation.permute(0, 2, 1)

        reshaped_input = input.view([-1, max_frames * self.groups, feature_size])
        vlad = activation.matmul(reshaped_input)
        vlad = vlad.permute(0, 2, 1)
        vlad = vlad - a
#         vlad = torch.div(vlad, vlad.norm(dim=1, keepdim=True))
        vlad = torch.nn.functional.normalize(vlad,dim=1)
        vlad = vlad.reshape([-1, self.nextvlad_cluster_size * feature_size])
        vlad = self.vlad_bn(vlad)
        return vlad

class SimpleHead(nn.Module):
    """Heads."""
    def __init__(self, out_size, dropout_prob=0.5, num_classes=82):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.clf = nn.Linear(out_size, num_classes)

    def forward(self, features):
        x = self.dropout(features)
        x = self.clf(x)
        return x

class Model(nn.Module):
    def __init__(self, config,video_embed_dim=1280, audio_embed_dim=128, hidden_dim=512,modal_dim=1024, image_model_name='efficientnet-b0',
                 num_classes=82, classifier_type='ClassificationHead', is_fusion=False,label_graph=None):
        super(Model, self).__init__()
        
        self.video_embed_dim = config['video_embed_dim'] #!!!
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.classifier_type = classifier_type
        self.is_fusion = is_fusion

        self.use_modal_drop = config['use_modal_drop']
        self.modal_drop_rate = config['modal_drop_rate']
        self.with_video_head = config['with_video_head']
        self.with_audio_head = config['with_audio_head']
        self.with_text_head = config['with_text_head']
        self.with_image_head = config['with_image_head']
        self.with_occur_head = config['with_occur_head']
        
        self.with_label_graph = config['with_label_graph']
        self.noise = config['use_noise']
        self.noise_prob = config['noise_prob']
        # 创建video模型
        video_dim = 0
        if self.with_video_head:
            self.video_model = NextVLAD(embed_dim=video_embed_dim, nextvlad_cluster_size=8,output_dim=modal_dim)  # 128
#             video_dim = 16384
            video_dim = self.video_embed_dim
        # 创建audio模型
        audio_dim = 0 
        if self.with_audio_head:
            self.audio_model = NextVLAD(embed_dim=audio_embed_dim, nextvlad_cluster_size=64,output_dim=modal_dim)  # 64
            audio_dim = 1024
#             audio_dim = 2048
        # 创建BERT模型，并且导入预训练模型
        text_out_dim = 0 
        if self.with_text_head:
            self.text_bert = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
            # 对text做fintnue
            self.text_bert.pooler.apply(self._init_weights)  
            self.text_linear = nn.Linear(self.text_bert.pooler.dense.out_features, modal_dim*2)
            self.text_bn = nn.BatchNorm1d(modal_dim*2)
            text_out_dim = modal_dim*2
        # 创建image模型
        image_out_dim = 0
#         if self.with_image_head:
#             self.image_model = EfficientNet.from_pretrained(image_model_name)
#             self.image_model._fc = nn.Linear(in_features=self.image_model._fc.in_features,out_features=modal_dim,bias=True)
#             image_out_dim = modal_dim
#             self.avg_pooling = nn.AdaptiveAvgPool2d(1)
#             self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # use vit to extract image feature in end to end manner
        if self.with_image_head:
            self.image_model = VITACExtractor()
            image_out_dim = 768
        
        out_size = video_dim + audio_dim + text_out_dim +  image_out_dim 
        if self.classifier_type == 'ClassificationHead':
            self.classifier = ClassificationHead(out_size=out_size, hidden_dim=self.hidden_dim, dropout_prob=0.5, num_classes=num_classes)
        self.classifier.apply(self._init_weights)
        
            
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.06)
            #nn.init.kaiming_normal_(module.weight)

    def _modal_drop(self,x,modal_name,rate=0.0):
        batch_size = x.shape[0]
        if modal_name == "video":
            drop_shape = torch.ones((batch_size, 1, 1))
        elif modal_name == "audio":
            drop_shape = torch.ones((batch_size, 1, 1))
        elif modal_name == "text":
            drop_shape = torch.ones((batch_size, 1))
        elif modal_name == "image":
            drop_shape = torch.ones((batch_size, 1, 1, 1))
        random_scale = torch.rand_like(drop_shape)
        keep_mask = random_scale > rate
        result = x * keep_mask.cuda()
        return result

            
    def forward(self, video_feature, audio_feature, text, image, image_only=False):
        """
        :param video_feature: shape=(b, num_frames, 1024)
        :param audio_feature: shape=(b, num_frames, 128)
        :param text: shape=(b, max_text_len), default max_text_len=128
        :param image: shape=(b, c, h, w)
        :return:
        """
        if self.training:
            if self.noise:
#                 video_noise = (torch.rand_like(video_feature)).cuda()
                video_noise = torch.normal(mean=0,std=1,size=video_feature.shape).cuda()
                video_noise[video_noise<self.noise_prob]=0
                video_feature += video_noise
#                 audio_noise = (torch.rand_like(audio_feature)).cuda()
                audio_noise = torch.normal(mean=0,std=1,size=audio_feature.shape).cuda()
                audio_noise[audio_noise<self.noise_prob]=0
                audio_feature += audio_noise
        outputs = []
        # video
        if self.with_video_head:
            if image_only:
                pass
            else:
                if self.use_modal_drop and self.training:
                    video_feature = self._modal_drop(video_feature,modal_name="video",rate=self.modal_drop_rate)
                video_embed = self.video_model(video_feature)
                if self.is_fusion:
                    video_embed = self.video_fusion(video_embed)
                outputs.append(video_embed)
        # audio
        if self.with_audio_head:
            if image_only:
                pass
            else:
                if self.use_modal_drop and self.training:
                    audio_feature = self._modal_drop(audio_feature,modal_name="audio",rate=self.modal_drop_rate)
                audio_embed = self.audio_model(audio_feature)
                if self.is_fusion:
                    audio_embed = self.audio_fusion(audio_embed)
                outputs.append(audio_embed)

        # text
        if self.with_text_head:
            if image_only:
                pass
            else:
                if self.use_modal_drop and self.training:
                    text = self._modal_drop(text,modal_name="text",rate=self.modal_drop_rate)
                hidden_states = self.text_bert(text)[1]
                text_emb = self.text_linear(hidden_states)
                text_emb = self.text_bn(text_emb)
                if self.is_fusion:
                    text_embed = self.text_fusion(torch.cat([text_embed_max, text_embed_mean], dim=1))
                    outputs.append(text_embed)
                else:
                    outputs.append(text_emb)
        # image
        if self.with_image_head:
            if self.use_modal_drop and self.training:
                image = self._modal_drop(image,modal_name="image",rate=self.modal_drop_rate)
#             image_emb = self.image_model(image)
            image_emb = self.image_model.extract_rgb_frame_features(image) 
            if self.is_fusion:
                image_embed = self.image_fusion(torch.cat([image_feature_max, image_feature_avg], dim=1))
                outputs.append(image_embed)
            else:
                outputs.append(image_emb)
        if image_only:
            image_output = image_emb
            image_output = self.simplehead(image_output)
            return image_output
        if self.with_occur_head:
            occur_feat = self.occur_linear(occur)
            outputs.append(occur_feat)
        logits = self.classifier(outputs)
        return logits