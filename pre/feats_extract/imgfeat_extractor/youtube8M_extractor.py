# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Facilitates extracting YouTube8M features from RGB images."""

import os
import sys
import tarfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
import torch
#INCEPTION_TF_GRAPH = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#YT8M_PCA_MAT = 'http://data.yt8m.org/yt8m_pca.tgz'
MODEL_DIR = os.path.join('pretrained', 'inception')
print(MODEL_DIR)


class YouTube8MFeatureExtractor(object):
  """Extracts YouTube8M features for RGB frames.

  First time constructing this class will create directory `yt8m` inside your
  home directory, and will download inception model (85 MB) and YouTube8M PCA
  matrix (15 MB). If you want to use another directory, then pass it to argument
  `model_dir` of constructor.

  If the model_dir exist and contains the necessary files, then files will be
  re-used without download.

  Usage Example:

      from PIL import Image
      import numpy

      # Instantiate extractor. Slow if called first time on your machine, as it
      # needs to download 100 MB.
      extractor = YouTube8MFeatureExtractor()

      image_file = os.path.join(extractor._model_dir, 'cropped_panda.jpg')

      im = np.array(Image.open(image_file))
      features = extractor.extract_rgb_frame_features(im)

  ** Note: OpenCV reverses the order of channels (i.e. orders channels as BGR
  instead of RGB). If you are using OpenCV, then you must do:

      im = im[:, :, ::-1]  # Reverses order on last (i.e. channel) dimension.

  then call `extractor.extract_rgb_frame_features(im)`
  """

  def __init__(self, model_dir=MODEL_DIR, use_batch=False):
    # Create MODEL_DIR if not created.
    self._model_dir = model_dir
    assert os.path.exists(model_dir)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_batch:
      inception_proto_file = os.path.join(self._model_dir, 'classify_image_graph_def_batch.pb')
    else:
      inception_proto_file = os.path.join(self._model_dir, 'classify_image_graph_def.pb')
    assert os.path.exists(inception_proto_file)
    self._load_inception(inception_proto_file)

    pca_mean = os.path.join(self._model_dir, 'mean.npy')
    if not os.path.exists(pca_mean):
      tarfile.open(download_path, 'r:gz').extractall(model_dir)
    self._load_pca()
    self._load_pca_gpu()

  def extract_rgb_frame_features(self, frame_rgb, apply_pca=True):
    """Applies the YouTube8M feature extraction over an RGB frame.

    This passes `frame_rgb` to inception3 model, extracting hidden layer
    activations and passing it to the YouTube8M PCA transformation.

    Args:
      frame_rgb: numpy array of uint8 with shape (height, width, channels) where
        channels must be 3 (RGB), and height and weight can be anything, as the
        inception model will resize.
      apply_pca: If not set, PCA transformation will be skipped.

    Returns:
      Output of inception from `frame_rgb` (2048-D) and optionally passed into
      YouTube8M PCA transformation (1024-D).
    """
    assert len(frame_rgb.shape) == 3
    assert frame_rgb.shape[2] == 3  # 3 channels (R, G, B)
    with self._inception_graph.as_default():
      frame_features = self.session.run('pool_3/_reshape:0',
                                        feed_dict={'DecodeJpeg:0': frame_rgb})
      frame_features = frame_features[0]  # Unbatch.
    if apply_pca:
      # frame_features = self.apply_pca(frame_features)
      frame_features = self.apply_pca_gpu(frame_features)

    return frame_features

  def extract_rgb_frame_features_list(self, frame_rgb_list, batch_size, apply_pca=True):
    input_list = []
    for _idx, frame_rgb in enumerate(frame_rgb_list):
      frame_rgb = np.expand_dims(frame_rgb, 0)
      if _idx % batch_size == 0:
        frame_rgb_batch = frame_rgb
      else:
        frame_rgb_batch = np.concatenate((frame_rgb_batch, frame_rgb), axis=0)
      if (_idx % batch_size == batch_size-1) or _idx == len(frame_rgb_list)-1:
        input_list.append(frame_rgb_batch)
    with self._inception_graph.as_default():
      frame_features_list = []
      for frame_rgb_batch in input_list:
        frame_features_batch = self.session.run('pool_3:0', feed_dict={'Placeholder_haoxin:0': frame_rgb_batch})
        frame_features_batch = frame_features_batch.reshape(-1, 2048)
        for _jdx in range(frame_features_batch.shape[0]):
          frame_features_list.append(frame_features_batch[_jdx, :])

    if apply_pca:
      frame_features_list = [self.apply_pca_gpu(frame_features) for frame_features in frame_features_list]
    return frame_features_list


  def apply_pca(self, frame_features):
    """Applies the YouTube8M PCA Transformation over `frame_features`.

    Args:
      frame_features: numpy array of floats, 2048 dimensional vector.

    Returns:
      1024 dimensional vector as a numpy array.
    """
    # Subtract mean
    feats = frame_features - self.pca_mean

    # Multiply by eigenvectors.
    feats = feats.reshape((1, 2048)).dot(self.pca_eigenvecs).reshape((1024,))

    # Whiten
    feats /= np.sqrt(self.pca_eigenvals + 1e-4)
    return feats

  def apply_pca_gpu(self, frame_features):
    """Applies the YouTube8M PCA Transformation over `frame_features`.

    Args:
      frame_features: numpy array of floats, 2048 dimensional vector.

    Returns:
      1024 dimensional vector as a numpy array.
    """
    frame_features = torch.from_numpy(frame_features).to(self.device)
    # Subtract mean
    feats = frame_features - self.pca_mean_gpu #(2048,)

    # Multiply by eigenvectors.
    feats = torch.mm(feats.reshape((1, 2048)),self.pca_eigenvecs_gpu).reshape((1024,))

    # Whiten
    feats /= torch.sqrt(self.pca_eigenvals_gpu + 1e-4)  # (1024,)
    feats = feats.cpu().numpy()
    return feats

  def _load_inception(self, proto_file):
    graph_def = tf.GraphDef.FromString(open(proto_file, 'rb').read())
    self._inception_graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)
    config.gpu_options.allow_growth = True
    with self._inception_graph.as_default():
      _ = tf.import_graph_def(graph_def, name='')
      self.session = tf.Session(config=config)

  def _load_pca(self):
    self.pca_mean = np.load(
        os.path.join(self._model_dir, 'mean.npy'))[:, 0]
    self.pca_eigenvals = np.load(
        os.path.join(self._model_dir, 'eigenvals.npy'))[:1024, 0]
    self.pca_eigenvecs = np.load(
        os.path.join(self._model_dir, 'eigenvecs.npy')).T[:, :1024]

  def _load_pca_gpu(self):
    self.pca_mean_gpu = torch.from_numpy(self.pca_mean).to(self.device)
    self.pca_eigenvals_gpu = torch.from_numpy(self.pca_eigenvals).to(self.device)
    self.pca_eigenvecs_gpu = torch.from_numpy(self.pca_eigenvecs).to(self.device)
