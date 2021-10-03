industry=$1    #meishi, 设置一级目录拼音
gpu=$2         #gpu 编号
tag_version=$3 #标签体系版本号
modal_num=$4
with_video=$5
imgfeat_extractor=$6 #图像特征提取模型 B16 L16 H14
audiofeat_extractor=$7 #音频特征提取模型
extract_type=$8
NUM_PROCESS=$9

if [ "$imgfeat_extractor" == "B16" ]; then
  imgfeat_extractor='VIT_B16'
fi
if [ "$imgfeat_extractor" == "L16" ]; then
  imgfeat_extractor='VIT_L16'
fi
if [ "$imgfeat_extractor" == "H14" ]; then
  imgfeat_extractor='VIT_H14'
fi
if [ "$audiofeat_extractor" == "" ]; then
  audiofeat_extractor='Vggish'
fi
if [ "$extract_type" == "" ]; then
  extract_type=1
fi
if [ "$NUM_PROCESS" == "" ]; then
  NUM_PROCESS=1
fi

VIDEO_DIR=../raw_data/tagging_dataset_test_5k_2nd  #视频存放目录
DATA_ROOT=../tagging/tagging_dataset_test_5k_2nd #特征存放目录

AUDIO_DIR=$DATA_ROOT/audios/${industry} #生成音频目录
IMAGE_DIR=$DATA_ROOT/image_jpg/${industry} #生成图片目录
AUDIO_NUMPY_DIR=$DATA_ROOT/audio_npy/${audiofeat_extractor}/${industry} #生成视频特征保存目录
VIDEO_NUMPY_DIR=${DATA_ROOT}/video_npy/${imgfeat_extractor}/${industry} #生成音频特征保存目录
TEXT_TXT_DIR=${DATA_ROOT}/text_txt/${industry} ##生成音频特征保存目录


IMAGE_BATCH_SIZE=1 #提图像特征batch size, 根据显存大小设置

FILE_DIR=${VIDEO_DIR}
postfix="mp4"

##1.2单进程提取特征
echo "特征提取中..."
export TF_FORCE_GPU_ALLOW_GROWTH=true
CUDA_VISIBLE_DEVICES=${gpu} python feat_extract_main.py --test_files_dir $FILE_DIR --frame_npy_folder ${VIDEO_NUMPY_DIR} --audio_npy_folder ${AUDIO_NUMPY_DIR} --image_batch_size ${IMAGE_BATCH_SIZE} --imgfeat_extractor ${imgfeat_extractor} --text_txt_folder ${TEXT_TXT_DIR} --image_jpg_folder ${IMAGE_DIR} --extract_type $extract_type --postfix $postfix

echo "特征提取中...FOR 可能遗漏的视频"
export TF_FORCE_GPU_ALLOW_GROWTH=true
CUDA_VISIBLE_DEVICES=${gpu} python feat_extract_main.py --test_files_dir $FILE_DIR --frame_npy_folder ${VIDEO_NUMPY_DIR} --audio_npy_folder ${AUDIO_NUMPY_DIR} --image_batch_size 2 --imgfeat_extractor ${imgfeat_extractor} --text_txt_folder ${TEXT_TXT_DIR} --image_jpg_folder ${IMAGE_DIR} --extract_type $extract_type --postfix $postfix
#1.2多进程提取特征(可能爆显存, 不推荐)
#for n in `seq 1 ${NUM_PROCESS}`
#do
#CUDA_VISIBLE_DEVICES=${gpu} python scripts/feat_extract_main.py --video_dir $VIDEO_DIR --frame_npy_folder ${VIDEO_NUMPY_DIR} --audio_npy_folder ${AUDIO_NUMPY_DIR} --image_batch_size ${IMAGE_BATCH_SIZE} --imgfeat_extractor ${imgfeat_extractor} --text_txt_folder ${TEXT_TXT_DIR} --image_jpg_folder ${IMAGE_DIR} --extract_type $extract_type --postfix $postfix &
#done
#wait

