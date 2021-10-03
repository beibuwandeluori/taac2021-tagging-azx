# coding=utf8
import os
import sys
import numpy as np
import tensorflow as tf

MODEL_DIR = os.path.join('pretrained', 'finetuned_resnet101')
print(MODEL_DIR)

class FinetunedResnet101Extractor(object):
  def __init__(self, model_dir=MODEL_DIR):
    self._model_dir = model_dir
    assert os.path.exists(model_dir)
    proto_file = os.path.join( self._model_dir, 'res_sku_new_3921_ocr_text_mix_710000_img_feature_Frozen.pb')
    assert os.path.exists(proto_file)
    self.load_model(proto_file)

  def extract_rgb_frame_features(self, frame_rgb):
    assert len(frame_rgb.shape) == 3
    assert frame_rgb.shape[2] == 3  # 3 channels (R, G, B)
    with self._graph.as_default():
      features = self.session.run(self.out_feature, feed_dict={self.input_img: frame_rgb, self.input_text_id: np.zeros((128))})
      features = features[0]  # Unbatch.
      assert features.shape[0] == 2816
      img_feat = features[0:2048]
    return img_feat

  def extract_rgb_frame_features_list(self, frame_rgb_list, batch_size):
    raise NotImplementedError("batch infer not supported error!!!")

  def load_model(self, proto_file):
    graph_def = tf.GraphDef.FromString(open(proto_file, 'rb').read())
    self._graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    with self._graph.as_default():
      _ = tf.import_graph_def(graph_def, name='')
      self.session = tf.Session(config=config)
    self.out_feature = self._graph.get_tensor_by_name('RecResults:0')
    self.input_img = self._graph.get_tensor_by_name('images:0')
    self.input_text_id = self._graph.get_tensor_by_name('texts:0')
