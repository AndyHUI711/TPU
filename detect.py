#!/usr/bin/python3

from os.path import join;
from math import ceil;
import numpy as np;
import cv2;
import tensorflow as tf;

class Encoder(object):

    def __init__(self, model_path = 'models'):

        self.model = tf.keras.models.load_model(join(model_path, 'facenet', '1'));

    def preprocess(self, img):

        assert img.shape[2] == 3;
        inputs = tf.expand_dims(img, axis = 0);
        inputs = tf.cast(inputs, dtype = tf.float32);
        mean = tf.math.reduce_mean(inputs, axis = [1,2,3]);
        std = tf.math.maximum(
            tf.math.sqrt(tf.math.reduce_mean(tf.math.pow(inputs - mean,2), axis = [1,2,3])),
            1.0 / tf.math.sqrt(tf.cast(img.size, dtype = tf.float32))
        );
        inputs = (inputs - mean) / std;
        outputs = tf.image.resize_with_pad(inputs, 160, 160);
        return outputs;

    def batch(self, imgs):

        inputs = [self.preprocess(img) for img in imgs];
        outputs = tf.concat(inputs, axis = 0);
        return outputs;

    def encode(self, imgs):

        assert type(imgs) is list;
        if len(imgs) == 0: return tf.zeros((0,self.model.signatures['serving_default'].outputs[0].shape[-1]), dtype = tf.float32);
        assert np.all([type(img) is np.ndarray and len(img.shape) == 3 for img in imgs]);
        batch = self.batch(imgs);
        feature = self.model.signatures['serving_default'](batch);
        feature = feature['Bottleneck_BatchNorm'];
        feature = feature / tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(feature, 2), axis = -1, keepdims = True));
        return feature;

if __name__ == "__main__":

    assert tf.executing_eagerly() == True;
    encoder = Encoder();