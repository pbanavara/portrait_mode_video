"""
model.py

Segmentation backbone networks.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)

Based on:
  - https://github.com/scaelles/OSVOS-TensorFlow/blob/master/osvos_parent_demo.py
    Written by Sergi Caelles (scaelles@vision.ee.ethz.ch)
    This file is part of the OSVOS paper presented in:
      Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
      One-Shot Video Object Segmentation
      CVPR 2017
    Unknown code license

References for future work:
    https://github.com/scaelles/OSVOS-TensorFlow
    http://localhost:8889/notebooks/models-master/research/slim/slim_walkthrough.ipynb
    https://github.com/bryanyzhu/two-stream-pytorch
    https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
    https://github.com/kwotsin/TensorFlow-ENet/blob/master/predict_segmentation.py
    https://github.com/fperazzi/davis-2017/tree/master/python/lib/davis/measures
    https://github.com/suyogduttjain/fusionseg
    https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, sys, warnings
import numpy as np
from datetime import datetime
from skimage.io import imsave
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tqdm import trange
slim = tf.contrib.slim
class Model(object):
    def __init__(self, checkpoint_file, segnet_stream='weak', config=None,
                 pred_masks_path='.', img_pred_masks_path='.'):
        if config is None:
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            self.config.allow_soft_placement = True
        tf.logging.set_verbosity(tf.logging.INFO)
    

        # Input data
        self.batch_size = 1
        if segnet_stream == 'weak':
            self.input_image = tf.placeholder(tf.float32, [self.batch_size, None, None, 4])
        else:
            self.input_image = tf.placeholder(tf.float32, [self.batch_size, None, None, 3])

        # Create the convnet
        with slim.arg_scope(self.backbone_arg_scope()):
            self.net, self.end_points = self.backbone(self.input_image, segnet_stream)
        self.probabilities = tf.nn.sigmoid(net)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Create a saver to load the network
        self.saver = tf.train.Saver([v for v in tf.global_variables() if '-up' not in v.name and '-cr' not in v.name])

        if not os.path.exists(pred_masks_path):
            os.makedirs(pred_masks_path)
        if not os.path.exists(img_pred_masks_path):
            os.makedirs(img_pred_masks_path)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(interp_surgery(tf.global_variables()))
        self.saver.restore(sess, checkpoint_file)
        
    def backbone_arg_scope(weight_decay=0.0002):
        """Defines the network's arg scope.
        Args:
            weight_decay: The l2 regularization coefficient.
        Returns:
            An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(stddev=0.001),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=None,
                            padding='SAME') as arg_sc:
            return arg_sc


    def crop_features(feature, out_size):
        """Crop the center of a feature map
        This is necessary when large upsampling results in a (width x height) size larger than the original input.
        Args:
            feature: Feature map to crop
            out_size: Size of the output feature map
        Returns:
            Tensor that performs the cropping
        """
        up_size = tf.shape(feature)
        ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
        ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
        slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
        # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
        return tf.reshape(slice_input, [int(feature.get_shape()[0]),
                                        out_size[1], out_size[2], int(feature.get_shape()[3])])


    def backbone(self, inputs, segnet_stream='weak'):
        """Defines the backbone network (same as the OSVOS network, with variation in input size)
        Args:
            inputs: Tensorflow placeholder that contains the input image (either 3 or 4 channels)
            segnet_stream: Is this the 3-channel or the 4-channel input version?
        Returns:
            net: Output Tensor of the network
            end_points: Dictionary with all Tensors of the network
        Reminder:
            
        """
        im_size = tf.shape(inputs)
        with tf.variable_scope(segnet_stream, segnet_stream, [inputs]) as sc:
            end_points_collection = sc.name + '_end_points'
            # Collect outputs of all intermediate layers.
            # Make sure convolution and max-pooling layers use SAME padding by default
            # Also, group all end points in the samqe container/collection
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                outputs_collections=end_points_collection):

                # VGG16 stage 1 has 2 convolution blocks followed by max-pooling
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')

                # VGG16 stage 2 has 2 convolution blocks followed by max-pooling
                net_2 = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net_2, [2, 2], scope='pool2')

                # VGG16 stage 3 has 3 convolution blocks followed by max-pooling
                net_3 = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net_3, [2, 2], scope='pool3')

                # VGG16 stage 4 has 3 convolution blocks followed by max-pooling
                net_4 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net_4, [2, 2], scope='pool4')

                # VGG16 stage 5 has 3 convolution blocks...
                net_5 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                # ...but here, it is not followed by max-pooling, as in the original VGG16 architecture.

                # This is where the specialization of the VGG network takes place, as described in DRIU and
                # OSVOS-S. The idea is to extract *side feature maps* and design *specialized layers* to perform
                # *deep supervision* targeted at a different task (here, segmentation) than the one used to
                # train the base network originally (i.e., large-scale natural image classification).

                # As explained in DRIU, each specialized side output produces feature maps in 16 different channels,
                # which are resized to the original image size and concatenated, creating a volume of fine-to-coarse
                # feature maps. one last convolutional layer linearly combines the feature maps from the volume
                # created by the specialized side outputs into a regressed result.  The convolutional layers employ
                # 3 x 3 convolutional filters for efficiency, except the ones used for linearly combining the outputs
                # (1 x 1 filters).

                with slim.arg_scope([slim.conv2d], activation_fn=None):

                    # Convolve last layer of stage 2 (before max-pooling) -> side_2 (None, 240, 427, 16)
                    side_2 = slim.conv2d(net_2, 16, [3, 3], scope='conv2_2_16')

                    # Convolve last layer of stage 3 (before max-pooling) -> side_3 (None, 120, 214, 16)
                    side_3 = slim.conv2d(net_3, 16, [3, 3], scope='conv3_3_16')

                    # Convolve last layer of stage 4 (before max-pooling) -> side_3 (None, 60, 117, 16)
                    side_4 = slim.conv2d(net_4, 16, [3, 3], scope='conv4_3_16')

                    # Convolve last layer of stage 3 (before max-pooling) -> side_3 (None, 30, 54, 16)
                    side_5 = slim.conv2d(net_5, 16, [3, 3], scope='conv5_3_16')

                    # The _S layears are the side output that will be used for deep supervision

                    # Dim reduction - linearly combine side_2 feature maps -> side_2_s (None, 240, 427, 1)
                    side_2_s = slim.conv2d(side_2, 1, [1, 1], scope='score-dsn_2')

                    # Dim reduction - linearly combine side_3 feature maps -> side_3_s (None, 120, 214, 1)
                    side_3_s = slim.conv2d(side_3, 1, [1, 1], scope='score-dsn_3')

                    # Dim reduction - linearly combine side_4 feature maps -> side_4_s (None, 60, 117, 1)
                    side_4_s = slim.conv2d(side_4, 1, [1, 1], scope='score-dsn_4')

                    # Dim reduction - linearly combine side_5 feature maps -> side_5_s (None, 30, 54, 1)
                    side_5_s = slim.conv2d(side_5, 1, [1, 1], scope='score-dsn_5')

                    # As repeated in OSVOS-S, upscaling operations take place wherever necessary, and feature
                    # maps from the separate paths are concatenated to construct a volume with information from
                    # different levels of detail. We linearly fuse the feature maps to a single output which has
                    # the same dimensions as the input image.
                    with slim.arg_scope([slim.convolution2d_transpose],
                                        activation_fn=None, biases_initializer=None, padding='VALID',
                                        outputs_collections=end_points_collection, trainable=False):

                        # Upsample the side outputs for deep supervision and center-cop them to the same size as
                        # the input. Note that this is straight upsampling (we're not trying to learn upsampling
                        # filters), hence the trainable=False param.

                        # Upsample side_2_s (None, 240, 427, 1) -> (None, 480, 854, 1)
                        # Center-crop (None, 480, 854, 1) to original image size (None, 480, 854, 1)
                        side_2_s = slim.convolution2d_transpose(side_2_s, 1, 4, 2, scope='score-dsn_2-up')
                        side_2_s = crop_features(side_2_s, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_2-cr', side_2_s)

                        # Upsample side_3_s (None, 120, 214, 1) -> (None, 484, 860, 1)
                        # Center-crop (None, 484, 860, 1) to original image size (None, 480, 854, 1)
                        side_3_s = slim.convolution2d_transpose(side_3_s, 1, 8, 4, scope='score-dsn_3-up')
                        side_3_s = crop_features(side_3_s, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_3-cr', side_3_s)

                        # Upsample side_4_s (None, 60, 117, 1) -> (None, 488, 864, 1)
                        # Center-crop (None, 488, 864, 1) to original image size (None, 480, 854, 1)
                        side_4_s = slim.convolution2d_transpose(side_4_s, 1, 16, 8, scope='score-dsn_4-up')
                        side_4_s = crop_features(side_4_s, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_4-cr', side_4_s)

                        # Upsample side_5_s (None, 30, 54, 1) -> (None, 496, 880, 1)
                        # Center-crop (None, 496, 880, 1) to original image size (None, 480, 854, 1)
                        side_5_s = slim.convolution2d_transpose(side_5_s, 1, 32, 16, scope='score-dsn_5-up')
                        side_5_s = crop_features(side_5_s, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/score-dsn_5-cr', side_5_s)

                        # Upsample the main outputs and center-cop them to the same size as the input
                        # Note that this is straight upsampling (we're not trying to learn upsampling filters),
                        # hence the trainable=False param. Then, concatenate thm in a big volume of fine-to-coarse
                        # feature maps of the same size.

                        # Upsample side_2 (None, 240, 427, 16) -> side_2_f (None, 480, 854, 16)
                        # Center-crop (None, 480, 854, 16) to original image size (None, 480, 854, 16)
                        side_2_f = slim.convolution2d_transpose(side_2, 16, 4, 2, scope='score-multi2-up')
                        side_2_f = crop_features(side_2_f, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi2-cr', side_2_f)

                        # Upsample side_2 (None, 120, 214, 16) -> side_2_f (None, 488, 864, 16)
                        # Center-crop (None, 488, 864, 16) to original image size (None, 480, 854, 16)
                        side_3_f = slim.convolution2d_transpose(side_3, 16, 8, 4, scope='score-multi3-up')
                        side_3_f = crop_features(side_3_f, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi3-cr', side_3_f)

                        # Upsample side_2 (None, 60, 117, 16) -> side_2_f (None, 488, 864, 16)
                        # Center-crop (None, 488, 864, 16) to original image size (None, 480, 854, 16)
                        side_4_f = slim.convolution2d_transpose(side_4, 16, 16, 8, scope='score-multi4-up')
                        side_4_f = crop_features(side_4_f, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi4-cr', side_4_f)

                        # Upsample side_2 (None, 30, 54, 16) -> side_2_f (None, 496, 880, 16)
                        # Center-crop (None, 496, 880, 16) to original image size (None, 480, 854, 16)
                        side_5_f = slim.convolution2d_transpose(side_5, 16, 32, 16, scope='score-multi5-up')
                        side_5_f = crop_features(side_5_f, im_size)
                        utils.collect_named_outputs(end_points_collection, segnet_stream + '/side-multi5-cr', side_5_f)

                    # Build the main volume concat_side (None, 496, 880, 16x4)
                    concat_side = tf.concat([side_2_f, side_3_f, side_4_f, side_5_f], axis=3)

                    # Dim reduction - linearly combine concat_side feature maps -> (None, 496, 880, 1)
                    net = slim.conv2d(concat_side, 1, [1, 1], scope='upscore-fuse')

                    # Note that the FC layers of the original VGG16 network are not part of the DRIU architecture

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


    def upsample_filt(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)


    # Set deconvolutional layers to compute bilinear interpolation
    def interp_surgery(variables):
        interp_tensors = []
        for v in variables:
            if '-up' in v.name:
                h, w, k, m = v.get_shape()
                tmp = np.zeros((m, k, h, w))
                if m != k:
                    raise ValueError('input + output channels need to be the same')
                if h != w:
                    raise ValueError('filters need to be square')
                up_filter = upsample_filt(int(h))
                tmp[range(m), range(k), :, :] = up_filter
                interp_tensors.append(tf.assign(v, tmp.transpose((2, 3, 1, 0)), validate_shape=True, use_locking=True))
        return interp_tensors

    # TODO: Move preprocessing to Tensorflow API?
    def preprocess_inputs(inputs, segnet_stream='weak'):
        """Preprocess the inputs to adapt them to the network requirements
        Args:
            Image we want to input to the network in (batch_size,W,H,3) or (batch_size,W,H,4) np array
        Returns:
            Image ready to input to the network with means substracted
        """
        assert(len(inputs.shape) == 4)

        if segnet_stream == 'weak':
            new_inputs = np.subtract(inputs.astype(np.float32), np.array((104.00699, 116.66877, 122.67892, 128.), dtype=np.float32))
        else:
            new_inputs = np.subtract(inputs.astype(np.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
        # input = tf.subtract(tf.cast(input, tf.float32), np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
        # input = np.expand_dims(input, axis=0)
        return new_inputs

    def test(self, image):
        """Test one sequence
        Args:
            dataset: Reference to a Dataset object instance
            checkpoint_path: Path of the checkpoint to use for the evaluation
            segnet_stream: Binary segmentation network stream; either "appearance stream" or 
            "flow stream" ['weak'|'full']
            pred_masks_path: Path to save the individual predicted masks
            img_pred_masks_path: Path to save the composite of the input image overlayed 
            with the predicted masks
            config: Reference to a Configuration object used in the creation of a Session
        Returns:
        """
        ret_arr = []
        rounds = 1
        #print("Output node naems ::: {}".format(output_node_names))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _round in trange(rounds, ascii=True, ncols=80, desc='Saving predictions as PNGs'):
                #samples, output_files = dataset.next_batch(batch_size, 'test', segnet_stream)
                inputs = self.preprocess_inputs(image, segnet_stream)
                masks = self.sess.run(probabilities, feed_dict={self.input_image: image})
                masks = np.where(masks.astype(np.float32) < 162.0/255.0, 0, 255).astype('uint8')
                for mask in masks:
                    #print(mask)
                    imsave(os.path.join(pred_masks_path, "output_file_2.png"), mask[:, :, 0])
                    ret_arr.append(mask[:, :, 0])
        return ret_arr

if __name__ == "__main__":
    segnet_stream = 'weak'
    ckpt_name = 'vgg_16_4chan_' + segnet_stream
    max_training_iters = 50000
    ckpt_path = '/home/ubuntu/dev/segment/weak_simple_model/' + ckpt_name + '/' + ckpt_name + '.ckpt-' + str(max_training_iters)
    model = Model(ckpt_path)
    x1 = 42
    y1 =  30
    x2 = x1 + 557
    y2 = y1+ 369
    bbox = [y1, x1, y2, x2]
    model = Model(ckpt_path)
