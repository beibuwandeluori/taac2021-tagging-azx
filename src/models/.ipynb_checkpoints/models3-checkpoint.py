

from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torchvision
import os
import torch
from transformers import BertTokenizer, BertModel
from network.fusion_head import FusionHead
from network.fusion_head_v2 import CrossFusionHead
from network.label_gcn import LabelGCN
def get_efficientnet(model_name='efficientnet-b0', num_classes=2):
    net = EfficientNet.from_pretrained(model_name)
    # net = EfficientNet.from_name(model_name)
    in_features = net._fc.in_features
    net._fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    return net
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=82):
        super().__init__()
#         self.norm = nn.BatchNorm1d(out_size)
        self.dense1 = nn.Linear(out_size, 1024)
        self.dense2 = nn.Linear(2048, 512)
        self.norm_0 = nn.BatchNorm1d(1024)

#         self.norm_1 = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.7)


        self.dense3 = nn.Linear(1024, 512)
        self.norm_2 = nn.BatchNorm1d(512)
        self.out_proj = nn.Linear(1024, num_classes)
#         self.out_proj = nn.Linear(out_size, num_classes)

    def forward(self, features):
        x = torch.cat(features, -1)
#         features = torch.cat(features, -1)
#         x = self.norm(features)
        # 因为这个dropout本质上是对多模态做了mask,去掉之后下降了,因此说明dropout有用.可以试下modal drop
    
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.relu(self.norm_0(x))
        x = self.dropout(x)
#         x = self.dense3(x)
#         x = torch.relu(self.norm_2(x))
#         x = self.dropout(x)
        x = self.out_proj(x)
        
        return x

# class ClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#     def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=82):
#         super().__init__()
# #         self.norm = nn.BatchNorm1d(out_size)
#         self.dense1 = nn.Linear(1024*5, 1024)
#         self.dense2 = nn.Linear((1280 + 1024 + 768)*8, 2048)
#         self.norm_0 = nn.BatchNorm1d(1024)

#         self.norm_1 = nn.BatchNorm1d(2048)
#         self.dropout = nn.Dropout(0.8)
#         self.dropout1 = nn.Dropout(0.6)


#         self.dense3 = nn.Linear(1024, 512)
#         self.norm_2 = nn.BatchNorm1d(512)
#         self.out_proj = nn.Linear(1024, num_classes)
# #         self.out_proj = nn.Linear(out_size, num_classes)
        

#     def forward(self, features):
#         x1 = torch.cat([features[0], features[1], features[2]], -1)
#         x1 = self.dropout1(x1)
#         x1 = self.dense2(x1)
#         x1 = torch.relu(self.norm_1(x1))

        
        
        
#         x = torch.cat([x1, features[3], features[4]], -1)
# #         features = torch.cat(features, -1)
# #         x = self.norm(features)
#         # 因为这个dropout本质上是对多模态做了mask,去掉之后下降了,因此说明dropout有用.可以试下modal drop
    
#         x = self.dropout(x)
#         x = self.dense1(x)
#         x = torch.relu(self.norm_0(x))
#         x = self.dropout(x)
# #         x = self.dense3(x)
# #         x = torch.relu(self.norm_2(x))
# #         x = self.dropout(x)
#         x = self.out_proj(x)
        
#         return x
    
# class ClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#     def __init__(self, out_size, hidden_dim=512, dropout_prob=0.5, num_classes=82):
#         super().__init__()
# #         self.norm = nn.BatchNorm1d(out_size)
#         self.dense1 = nn.Linear(1024 * 6, 1024)
#         self.dense2 = nn.Linear(1280*8, 1024)
#         self.dense3 = nn.Linear(4096 * 2, 1024)

#         self.norm_0 = nn.BatchNorm1d(1024)

# #         self.norm_1 = nn.BatchNorm1d(1024)
# #         self.dropout1 = nn.Dropout(0.)
#         self.dropout = nn.Dropout(0.7)


# #         self.dense3 = nn.Linear(1024, 512)
#         self.norm_2 = nn.BatchNorm1d(2048)
#         self.out_proj = nn.Linear(1024, num_classes)
# #         self.out_proj = nn.Linear(out_size, num_classes)

#     def forward(self, features):
#         xv = self.dropout(features[0])
#         xv = self.dense2(xv)        
#         xv = torch.relu(self.norm_0(xv))
        
#         xv1 = self.dropout(features[1])
#         xv1 = self.dense3(xv1)        
#         xv1 = torch.relu(self.norm_0(xv1))
#         x = torch.cat([xv, xv1, features[2], features[3], features[4]], -1)
# #         features = torch.cat(features, -1)
# #         x = self.norm(features)
#         # 因为这个dropout本质上是对多模态做了mask,去掉之后下降了,因此说明dropout有用.可以试下modal drop
    
#         x = self.dropout(x)
#         x = self.dense1(x)
#         x = torch.relu(self.norm_0(x))
#         x = self.dropout(x)
# #         x = self.dense3(x)
# #         x = torch.relu(self.norm_2(x))
# #         x = self.dropout(x)
#         x = self.out_proj(x)
        
#         return x

class SEFusion(nn.Module):
    """Dropout + Channel Attention
    """
    def __init__(self, feat_dim, hidden1_size=1024, drop_rate=0.5, gating_reduction=8, num_classes=82,gating_last_bn=False):
        super(SEFusion, self).__init__()
        self.dropout = nn.Dropout(drop_rate)  # dropout训练
        self.hidden1_size = hidden1_size
        self.gating_reduction = gating_reduction
        self.gating_last_bn = gating_last_bn
        self.linear = nn.Linear(feat_dim, self.hidden1_size)
        self.linear_bn = nn.BatchNorm1d(self.hidden1_size)
        self.gating_linear = nn.Linear(self.hidden1_size, self.hidden1_size // self.gating_reduction)
        self.gating_bn = nn.BatchNorm1d(self.hidden1_size // self.gating_reduction)
        self.gating_linear2 = nn.Linear(self.hidden1_size // self.gating_reduction, self.hidden1_size)
        self.gating_last_bn = gating_last_bn
        self.last_bn = nn.BatchNorm1d(self.hidden1_size)
        self.out_proj = nn.Linear(self.hidden1_size, num_classes)

    def forward(self, input_list):
        concat_feat = torch.cat(input_list, 1)
        concat_feat = self.dropout(concat_feat)
        activation = self.linear(concat_feat)
        activation = self.linear_bn(activation)
        gates = self.gating_linear(activation)
        gates = torch.relu(self.gating_bn(gates))
        gates = self.gating_linear2(gates)
        if self.gating_last_bn:
            gates = self.last_bn(gates)

        gates = torch.sigmoid(gates)
        # bias = False
        activation = activation * gates

        activation = self.dropout(activation)
        activation = self.out_proj(activation)
        return activation


class MoeModel(nn.Module):
    def __init__(self, input_size, num_classes,hidden_dim=512, num_mixtures=4, l2_penalty=0.0):
        super(MoeModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_mixtures = num_mixtures
        self.l2_penalty = l2_penalty        
        
#         self.norm = nn.BatchNorm1d(self.input_size)
#         self.dense = nn.Linear(self.input_size, hidden_dim)
#         self.norm_1 = nn.BatchNorm1d(hidden_dim)
#         self.dropout = nn.Dropout(0.5)
#         self.dense_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.norm_2 = nn.BatchNorm1d(hidden_dim)
        
#       # TODO 加上对以下两个层的L2正则惩罚
#         self.gate_linear = nn.Linear(self.input_size, self.num_classes * (self.num_mixtures+1))  # L2
#         self.expert_linear = nn.Linear(self.input_size, self.num_classes * self.num_mixtures)  # L2
        self.gate_linear = nn.Linear(hidden_dim, self.num_classes * (self.num_mixtures+1))  # L2
        self.expert_linear = nn.Linear(hidden_dim, self.num_classes * self.num_mixtures)  # L2


    def forward(self, model_input):
#         x = self.norm(model_input)
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.relu(self.norm_1(x))
#         x = self.dropout(x)
#         x = self.dense_1(x)
#         x = torch.relu(self.norm_2(x))
#         model_input = self.dropout(x)
        gate_activations = self.gate_linear(model_input)
        expert_activations = self.expert_linear(model_input)
        gating_distribution = torch.softmax(gate_activations.view(-1, self.num_mixtures+1), dim=1)  # (Batch * #Labels) x (num_mixtures + 1)
        # TODO 恢复原样
        expert_distribution = torch.sigmoid(expert_activations.view(-1, self.num_mixtures))  # (Batch * #Labels) x num_mixtures
        #expert_distribution = expert_activations.view(-1, self.num_mixtures)  # (Batch * #Labels) x num_mixtures
        final_probabilities = torch.sum(gating_distribution[:, :self.num_mixtures] * expert_distribution, 1).view(-1, self.num_classes)
        return final_probabilities


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
#         res = input
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
    def __init__(self, config, video_embed_dim=1280, video_embed_dim2=768, video_embed_dim3=1024, audio_embed_dim=128, hidden_dim=512,modal_dim=1024, image_model_name='efficientnet-b0',
                 num_classes=82, classifier_type='ClassificationHead', is_fusion=False,label_graph=None):
        super(Model, self).__init__()
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
            self.video_model = NextVLAD(embed_dim=video_embed_dim, nextvlad_cluster_size=8*2,output_dim=modal_dim)  # 128
            video_dim = 1280*2
            self.video_model2 = NextVLAD(embed_dim=video_embed_dim2, nextvlad_cluster_size=8,output_dim=modal_dim)  # 128
            video_dim2 = 768
            self.video_model3 = NextVLAD(embed_dim=video_embed_dim3, nextvlad_cluster_size=8*4,output_dim=modal_dim)  # 128
            video_dim3 = 1024*4

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
#             self.text_bert = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
            # 对text做fintnue
            self.text_bert.pooler.apply(self._init_weights)  
#             self.text_bert.encoder.layer[11].apply(self._init_weights)
#             self.text_bert.encoder.layer[10].apply(self._init_weights)
#             text_out_dim = self.text_bert.pooler.dense.out_features
            self.text_linear = nn.Linear(self.text_bert.pooler.dense.out_features, modal_dim*2)
            self.text_bn = nn.BatchNorm1d(modal_dim*2)
            text_out_dim = modal_dim*2
        # 创建image模型
        image_out_dim = 0
        if self.with_image_head:
            self.image_model = EfficientNet.from_pretrained(image_model_name)
            self.image_model._fc = nn.Linear(in_features=self.image_model._fc.in_features,out_features=modal_dim,bias=True)
            image_out_dim = modal_dim
            self.avg_pooling = nn.AdaptiveAvgPool2d(1)
            self.max_pooling = nn.AdaptiveMaxPool2d(1)
        occur_dim = 0
        if self.with_occur_head:
            occur_dim = 1024
            self.occur_linear = nn.Linear(768,occur_dim)
        
        label_dim = 0
        if self.with_label_graph:
            label_dim = 512
            self.labelGCN = LabelGCN(node_feature_size=label_dim,hidden_size=label_dim,num_classes=82)
            #self.graph_classifier= SimpleHead(out_size=label_dim, num_classes=num_classes)

        # 创建fusion分类器，随机初始化
        # out_size = video_embed_dim//2 + audio_embed_dim//2 + (text_out_dim + image_out_dim) * 2
        if self.is_fusion:
            self.video_fusion = SEFusion(16384, hidden1_size=1024, drop_rate=0.8, gating_reduction=8, gating_last_bn=False)
            self.audio_fusion = SEFusion(1024, hidden1_size=1024, drop_rate=0.8, gating_reduction=8, gating_last_bn=False)
            self.text_fusion = SEFusion(text_out_dim * 2, hidden1_size=1024, drop_rate=0.8, gating_reduction=8, gating_last_bn=False)
            self.image_fusion = SEFusion(image_out_dim * 2, hidden1_size=1024, drop_rate=0.8, gating_reduction=8, gating_last_bn=False)
            out_size = 1024 * 4
        else:
            out_size = video_dim + video_dim2 +video_dim3 +  audio_dim + text_out_dim +  image_out_dim + occur_dim 
#             out_size = video_dim + audio_dim + text_out_dim +  ( image_out_dim) * 2 + occur_dim
#             out_size = modal_dim * 4
#             self.fusion_layer = SEFusion(out_size, hidden1_size=512, drop_rate=0.5, gating_reduction=8, gating_last_bn=False)
        if self.classifier_type == 'ClassificationHead':
            self.classifier = ClassificationHead(out_size=out_size, hidden_dim=self.hidden_dim, dropout_prob=0.5, num_classes=num_classes)#hidden_dim=self.hidden_dim
        elif self.classifier_type == "TransformerHead":
            self.classifier = FusionHead(hidden=modal_dim,n_layers=1,attn_heads=16, dropout=0.5, num_classes=num_classes)            
        elif self.classifier_type == "CrossTransformerHead":
            self.classifier = CrossFusionHead(modal_num=4,hidden=modal_dim,attn_heads=1, dropout=0.5, num_classes=num_classes)            
        else:  # MoeModel
            print(11)
            self.classifier = MoeModel(input_size=out_size, num_classes=num_classes, num_mixtures=4, l2_penalty=0.0)
        self.classifier.apply(self._init_weights)
#         self.simplehead = SimpleHead(image_out_dim, dropout_prob=0.5, num_classes=82)
        
            
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

            
    def forward(self, video_feature, video_feature2,video_feature3, audio_feature, text, image, image_only=False):
        """
        :param video_feature: shape=(b, num_frames, 1024)
        :param audio_feature: shape=(b, num_frames, 128)
        :param text: shape=(b, max_text_len), default max_text_len=128
        :param image: shape=(b, c, h, w)
        :return:
        """
        image_only = False
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
                    video_feature2 = self._modal_drop(video_feature2,modal_name="video",rate=self.modal_drop_rate)
                    video_feature3 = self._modal_drop(video_feature3,modal_name="video",rate=self.modal_drop_rate)
#                     video_feature4 = self._modal_drop(video_feature4,modal_name="video",rate=self.modal_drop_rate)

                video_embed = self.video_model(video_feature)
                video_embed2 = self.video_model2(video_feature2)
                video_embed3 = self.video_model3(video_feature3)
#                 video_embed4 = self.video_model4(video_feature4)

                if self.is_fusion:
                    video_embed = self.video_fusion(video_embed)
                outputs.append(video_embed)
                outputs.append(video_embed2)
                outputs.append(video_embed3)
#                 outputs.append(video_embed4)
                
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
    #             layer = 3
    #             hidden_states = self.text_bert(text,output_hidden_states=True)[2][layer-1] # 第layer层输出
    #             text_emb = hidden_states[:,0]
    #             text_emb = self.text_linear(text_emb)
                hidden_states = self.text_bert(text)[1]
                text_emb = self.text_linear(hidden_states)
                text_emb = self.text_bn(text_emb)

    #             text_embed_max = hidden_states.max(1)[0].float()
    #             text_embed_mean = hidden_states.mean(1).float()
                if self.is_fusion:
                    text_embed = self.text_fusion(torch.cat([text_embed_max, text_embed_mean], dim=1))
                    outputs.append(text_embed)
                else:
                    outputs.append(text_emb)
    #                 outputs.append(text_embed_max)
    #                 outputs.append(text_embed_mean)

        # image
        if self.with_image_head:
            if self.use_modal_drop and self.training:
                image = self._modal_drop(image,modal_name="image",rate=self.modal_drop_rate)
            image_emb = self.image_model(image)
#             image_feature = self.image_model.extract_features(image)
#             image_feature_max = self.max_pooling(image_feature)
#             image_feature_max = image_feature_max.view(image_feature_max.size()[0], -1)
#             image_feature_avg = self.avg_pooling(image_feature)
#             image_feature_avg = image_feature_avg.view(image_feature_avg.size()[0], -1)
            if self.is_fusion:
                image_embed = self.image_fusion(torch.cat([image_feature_max, image_feature_avg], dim=1))
                outputs.append(image_embed)
            else:
                outputs.append(image_emb)
#                 outputs.append(image_feature_max)
#                 outputs.append(image_feature_avg)
        if image_only:
            image_output = image_emb
            image_output = self.simplehead(image_output)
            return image_output
        if self.with_occur_head:
            occur_feat = self.occur_linear(occur)
            outputs.append(occur_feat)
#         logits = self.fusion_layer(outputs)
        logits = self.classifier(outputs)
#         final_feat = self.classifier(outputs)
            
#         if self.with_label_graph:
#             label_feat = self.labelGCN(label_graph)
#             label_feat = label_feat.permute(1,0)
#             logits = final_feat.matmul(label_feat)
#             #label_logits = torch.cat([x.view(-1,1,self.modal_dim) x outputs],-1).matmul(label_feat)
#             return logits
            
#         print(video_embed.shape,audio_embed.shape,text_emb.shape,image_emb.shape)
        # 将特征输入分类器，得到82分类的logit
#         final_hidden_state = torch.cat(outputs, -1)
#         logits = self.classifier(final_hidden_state)

        #final_hidden_state = self.fusion_layer(final_hidden_state)
        # print(final_hidden_state.shape, logits.shape)
        #image_output = image_emb
        #image_output = self.simplehead(image_output)
        return logits
        #return logits, image_output


if __name__ == '__main__':
    # network = get_efficientnet('efficientnet-b0')
    # image_size = 224
    # network = network.to(torch.device('cpu'))
    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(network, input_s, device='cpu'))

    # m = nn.Dropout(p=0.5)
    # random_scale = torch.rand((20,1))
    # input = torch.randn(20, 16)
    # keep_mask = (random_scale>=0.5).type_as(input)
    # result = input * keep_mask
    # idx = m(torch.ones(20,1))
    # print(idx)
    # output = m(input)

    # model = Model(hidden_dim=1024, classifier_type='SEFusion', is_fusion=True)
    model = ModelV2()
    video = torch.randn(2, 120, 1024)  # your high resolution picture
    audio = torch.randn(2, 120, 128)  # your high resolution picture
    ids = torch.tensor([[101, 872, 1962, 8013, 102, 102], [101, 872, 1962, 8013, 102, 102]])
    img = torch.randn(2, 3, 224, 224)  # your high resolution picture
    # results = model(video, audio, ids, img)
    results, video_logits, audio_logits, text_logits, image_logits = model(video, audio, ids, img)
    # results = model(video, audio, ids, img, is_training=True)
    print(results.shape)
    # print(results[0].shape, results[1].shape, results[2].shape, results[3].shape, results[4].shape)

    # nextvlad_audio = NextVLAD(embed_dim=128, nextvlad_cluster_size=12)
    # nextvlad_video = NextVLAD(embed_dim=1024, nextvlad_cluster_size=12)
    # video = torch.randn(2, 300, 1024)  # your high resolution picture
    # audio = torch.randn(2, 300, 128)  # your high resolution picture
    # print(nextvlad_audio(audio).shape)
    # print(nextvlad_video(video).shape)

    # print(network._modules.items())
    # print(network)
    # tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    # network = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
    # # print(network)
    # ids = torch.tensor([[101,  872, 1962, 8013,  102], [101,  872, 1962, 8013,  102]])
    # print(ids.shape)
    # # inputs = tokenizer("你好！", return_tensors="pt")
    # # print(inputs, type(inputs))
    # # outputs = network(**inputs)[0]
    # outputs = network(ids)[0]
    # print(outputs.shape)

    pass