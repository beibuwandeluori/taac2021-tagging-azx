from __future__ import unicode_literals
import sys,os
import numpy as np
import cv2
import time
import tensorflow as tf
import json
import torch
from pre.feats_extract.imgfeat_extractor.youtube8M_extractor import YouTube8MFeatureExtractor
from pre.feats_extract.imgfeat_extractor.finetuned_resnet101 import FinetunedResnet101Extractor
from pre.feats_extract.imgfeat_extractor.VIT_B16 import VITB16ACExtractor
from pre.feats_extract.txt_extractor.text_requests import VideoASR,VideoOCR,ImageOCR
from pre.feats_extract.audio_extractor import vggish_input,vggish_params,vggish_postprocess,vggish_slim
from torchvision import transforms
from PIL import Image

FRAMES_PER_SECOND = 1
PCA_PARAMS = "pretrained/vggfish/vggish_pca_params.npz" #'Path to the VGGish PCA parameters file.'
VGGISH_CHECKPOINT = 'pretrained/vggfish/vggish_model.ckpt'
CAP_PROP_POS_MSEC = 0

class MultiModalFeatureExtract(object):
    """docstring for ClassName"""
    def __init__(self, batch_size = 1, 
                 imgfeat_extractor = 'Youtube8M',
                 extract_video = True,
                 extract_audio = True,
                 extract_text = True):
        super(MultiModalFeatureExtract, self).__init__()
        self.extract_video = extract_video
        self.extract_audio = extract_audio
        self.extract_text = extract_text
        
        #Video Extract
        if extract_video:
            self.batch_size = batch_size
            if imgfeat_extractor == 'Youtube8M':
                self.extractor = YouTube8MFeatureExtractor(use_batch = batch_size!=1)
            elif imgfeat_extractor == 'FinetunedResnet101':
                self.extractor = FinetunedResnet101Extractor()
            elif imgfeat_extractor == 'VIT-B/16-AC':
                self.extractor = VITB16ACExtractor()
                self.extractor.eval()
            else:
                raise NotImplementedError(imgfeat_extractor)
                
        if extract_audio:
            self.pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)  # audio pca
            self.audio_graph = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=True)
            config.gpu_options.allow_growth = True
            with self.audio_graph.as_default():
                # 音频
                self.audio_sess = tf.Session(graph=self.audio_graph, config=config)
                vggish_slim.define_vggish_slim(training=False)
                vggish_slim.load_vggish_slim_checkpoint(self.audio_sess, VGGISH_CHECKPOINT)
            self.features_tensor = self.audio_sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.audio_sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
            
        if extract_text:
            self.video_ocr_extractor = VideoOCR()
            self.video_asr_extractor = VideoASR()
            self.image_ocr_extractor = ImageOCR()

    def frame_iterator(self, filename, every_ms=1000, max_num_frames=300):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
          filename: Path to video file (e.g. mp4)
          every_ms: The duration (in milliseconds) to skip between frames.
          max_num_frames: Maximum number of frames to process, taken from the
            beginning of the video.

        Yields:
          RGB frame with shape (image height, image width, channels)
        """
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        while num_retrieved < max_num_frames:
          # Skip frames
          while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
            if not video_capture.read()[0]:
              return

          last_ts = video_capture.get(CAP_PROP_POS_MSEC)
          has_frames, frame = video_capture.read()
          if not has_frames:
            break
          yield frame
          num_retrieved += 1
    def frame_iterator_list(self, filename, every_ms=1000, max_num_frames=300):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
          print(sys.stderr, 'Error: Cannot open video file ' + filename)
          return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        frame_all = []
        while num_retrieved < max_num_frames:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return frame_all

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            frame_all.append(frame[:, :, ::-1])
            num_retrieved += 1

        return frame_all

    def extract_feat(self, test_file,
                     frame_npy_path=None, audio_npy_path=None, txt_file_path=None,
                     image_jpg_path=None, save=True):
        filetype = test_file.split('.')[-1]
        if filetype in ['mp4', 'avi']:
            feat_dict = self.extract_video_feat(test_file, frame_npy_path, audio_npy_path, txt_file_path,
                     image_jpg_path, save)
        elif filetype in ['jpg', 'png']:
            feat_dict = self.extract_image_feat(test_file)
        else:
            raise NotImplementedError
        if save:
            if 'video' in feat_dict:
                np.save(frame_npy_path, feat_dict['video'])
                print('保存视频特征为{}'.format(frame_npy_path))
            if 'audio' in feat_dict:
                np.save(audio_npy_path, feat_dict['audio'])
                print('保存音频特征为{}'.format(audio_npy_path))
            if 'text' in feat_dict:
                with open(txt_file_path, 'w') as f:
                    f.write(feat_dict['text'])
                print('保存文本特征为{}'.format(txt_file_path))
            if 'image' in feat_dict and filetype=='mp4':
                cv2.imwrite(image_jpg_path, feat_dict['image'][:,:,::-1])
        return feat_dict

    def extract_image_feat(self, test_file):
        feat_dict={}
        feat_dict['image'] = cv2.imread(test_file,1)[:,:,::-1] #convert to rgb

        if self.extract_text:
            start_time = time.time()
            image_ocr = self.image_ocr_extractor.request(test_file)        
            feat_dict['text'] = json.dumps({'image_ocr': image_ocr}, ensure_ascii=False)
            end_time = time.time()
            print("text extract cost {} sec".format(end_time - start_time))   
        return feat_dict
    
    def extract_video_feat(self, test_file,
                     frame_npy_path=None, audio_npy_path=None, txt_file_path=None,
                     image_jpg_path=None, save=True):
        feat_dict={}
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        #=============================================视频
        if (frame_npy_path is None or os.path.exists(frame_npy_path)) and save==True:
            pass
        else:
            start_time = time.time()
            if self.batch_size == 2:
                features_arr = []
                for rgb in self.frame_iterator(test_file, every_ms=1000.0/FRAMES_PER_SECOND):
                    img = Image.fromarray(rgb[:, :, ::-1].copy().astype('uint8')).convert('RGB')
                    features = self.extractor.extract_rgb_frame_features(transform_train(img).view(1,3,224,224)).squeeze().detach().numpy()
                    #features = self.extractor.extract_rgb_frame_features(torch.from_numpy(rgb[:, :, ::-1].copy()))
                    features_arr.append(features)
                feat_dict['video'] = features_arr
            else:
                rgb_list = self.frame_iterator_list(test_file, every_ms=1000.0/FRAMES_PER_SECOND)
                real_rgb_list = []
                for rgb in rgb_list:
                    img = Image.fromarray(rgb[:, :, ::-1].copy().astype('uint8')).convert('RGB')
                    real_rgb_list.append(transform_train(img))
#                     .view(1,3,224,224)
                real_rgb_list = torch.stack(real_rgb_list, 0)
#                 print(real_rgb_list.shape)
                    #features = self.extractor.extract_rgb_frame_features().squeeze().detach().numpy()
                feat_dict['video'] = self.extractor.extract_rgb_frame_features(real_rgb_list).detach().numpy()
            end_time = time.time()
            print("video extract cost {} sec".format(end_time - start_time))   
        #=============================================图片抽帧
        if (image_jpg_path is None or os.path.exists(image_jpg_path)) and save==True:
            pass
        else:
            start_time = time.time()
            rgb_list = self.frame_iterator_list(test_file, every_ms=1000.0/FRAMES_PER_SECOND)
            feat_dict['image'] = rgb_list[len(rgb_list)//2]
            end_time = time.time()
            print("image extract cost {} sec".format(end_time - start_time))
        #=============================================音频
        if (audio_npy_path is None or os.path.exists(audio_npy_path)) and save==True :
            #postprocessed_batch = np.load(audio_npy_path)
            pass
        else:
            start_time = time.time()
            output_audio = test_file.replace('.mp4','.wav')
            if not os.path.exists(output_audio):
                command = 'ffmpeg -loglevel error -i '+test_file+' '+output_audio
                os.system(command)
                #print("audio file not exists: {}".format(output_audio))
                #return
            examples_batch = vggish_input.wavfile_to_examples(output_audio)
            [embedding_batch] = self.audio_sess.run([self.embedding_tensor],
                                         feed_dict={self.features_tensor: examples_batch})
            feat_dict['audio'] = self.pproc.postprocess(embedding_batch)
            end_time = time.time()
            print("audio extract cost {} sec".format(end_time - start_time)) 
        #=============================================文本
        if (txt_file_path is None or os.path.exists(txt_file_path)) and save==True:
            pass
        elif self.extract_text:
            start_time = time.time()
            video_ocr = self.video_ocr_extractor.request(test_file)
            video_asr = self.video_asr_extractor.request(test_file)
            feat_dict['text'] = json.dumps({'video_ocr': video_ocr, 'video_asr': video_asr}, ensure_ascii=False)
            print(feat_dict['text'])
            end_time = time.time()
            print("text extract cost {} sec".format(end_time - start_time))
        return feat_dict
