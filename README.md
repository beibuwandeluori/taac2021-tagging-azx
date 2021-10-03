# 2021腾讯广告推荐大赛解决方案
## 队伍介绍
腾讯广告推荐大赛赛道二-第10名(最终获奖排名，线上排名14),[比赛链接:](https://algo.qq.com/index.html)
- 队名：凹造型（深圳大学计算机与软件学院队伍）  
- 成员：何铭凯(team leader)、骆锦潍、章涵艺、陈保营和胡彦杰

## 代码说明
- 有问题请联系我们队伍成员
- 网盘链接: https://pan.baidu.com/s/1Fe_y4TIsptqc1XVrFJu4og 提取码: utq9
- (B16特征与模型没用到 可以忽略)
- 链接中目录 “特征上传” 包含预提取的训练集特征与测试集特征（测试集可自己提，但我们也一并提供）
- 链接中目录 “特征提取模型上传” 包含视频特征提取模型的checkpoint*2 (我们使用两个视频特征提取模型)
- 注：L16提取时长大概4.5小时；H14提取时长大概8.5小时
    
# 运行步骤
### step1: 数据配置和环境配置
- 1.把复赛测试集原始视频放到这个目录:raw_data/tagging_dataset_test_5k_2nd （为了给你们重新提取测试集特征）

- 2.把训练集特征（包括文本和图片）放到目录:tagging/tagging_dataset_train_5k中四个模态对应目录
(audio,text,image都是baseline原本提取的特征,video存放我们提取的特征)
其中video_npy下两个文件夹分别存放网盘链接下的两个特征（H14,L16）中
如tagging/tagging_dataset_train_5k/videp_npy/VIT_H14/tagging中放5000个.npy文件（H14）

- 3.运行init.sh,安装requirement.txt中的环境(包括特征提取环境和模型运行环境)

### step2: 模型训练
- 运行train.sh,依次完成model1,model2,model3的10折训练（model3使用快照集成，10折中每一折选取最高的top2个pkl文件）,产生对应的模型参数到output/weight/{name}/中

### step3: 提取测试集视频特征

- 在网盘链接下载特征提取的模型checkpoint文件，
包括imagenet21k_ViT-H_14.npz, imagenet21k+imagenet2012_ViT-L_16-224.npz
放到这个目录: pre/feats_extract/imgfeat_extractor/checkpoint
依次执行run_feature_extract.ipynb中的命令 生成两种视频特征

- 注：把测试集特征（包括文本和图片）放到目录:tagging/tagging_dataset_test_5k_2nd中四个模态对应目录
(audio,text,image都是baseline原本提取的特征,video存放我们提取的特征)
其中video_npy下两个文件夹分别存放网盘链接下的两个特征（H14,L16）中
（如果你们要重新提取 麻烦确认特征有没有生成到指定文件夹，指定数目）
如tagging/tagging_dataset_test_5k_2nd/videp_npy/VIT_H14/tagging中放5000个.npy文件（H14）


### step4: 运行inference.sh
- 生成每一折的预测文件到output/pkl中
- 生成最终提交文件到output/result中

