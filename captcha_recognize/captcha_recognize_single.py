#!/usr/bin/env python
# -*- coding: utf-8 -*-
# FileName  : captcha_eval_single.py
# Author    : leafsummer@

import os
import sys
import config
import numpy as np
import tensorflow as tf
import captcha_model
from PIL import Image
from tensorflow import gfile

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT

CHAR_SETS = config.CHAR_SETS
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM


def one_hot_to_text(recog_result):
    char_list = []
    for i in recog_result:
        char_list.append(CHAR_SETS[i])
    text = ''.join(char_list)
    return text


def input_data(image_path):
    if not gfile.Exists(image_path):
        print("Image path: %s is not found" % image_path)
        return None

    images = np.zeros([1, IMAGE_HEIGHT*IMAGE_WIDTH], dtype='float32')
    files = []
    image = Image.open(image_path)
    image_gray = image.convert('L')
    image_resize = image_gray.resize(size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    image.close()
    input_img = np.array(image_resize, dtype='float32')
    input_img = np.multiply(input_img.flatten(), 1./255) - 0.5
    images[0,:] = input_img
    base_name = os.path.basename(image_path)
    files.append(base_name)

    return images, files


def run_predict(image_path):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        images, files = input_data(image_path)
        images = tf.constant(images)
        logits = captcha_model.inference(images, keep_prob=1)
        result = captcha_model.output(logits)
        saver = tf.train.Saver()
        sess = tf.Session()
        print(tf.train.latest_checkpoint('./models'))
        saver.restore(sess, tf.train.latest_checkpoint('./models'))
        recog_result = sess.run(result)
        sess.close()
        print("recog_result: %s" % recog_result)
        text = one_hot_to_text(recog_result[0])
        print("recog text is: %s" % text)
    return text


if __name__ == '__main__':
    # python captcha_recognize_single.py /tmp/captcha.png
    run_predict(sys.argv[1])