from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import os
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from captcha.image import ImageCaptcha

import config

IMAGE_HEIGHT = config.IMAGE_HEIGHT
IMAGE_WIDTH = config.IMAGE_WIDTH
CHARS_NUM = config.CHARS_NUM

TEST_SIZE = 1000
TRAIN_SIZE = 50000
VALID_SIZE = 20000

FLAGS = None


def generate_captcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, font_size=16, chars_num=4):
    dark_colors = ["black", "darkred", "darkgreen", "brown",
                "darkblue", "purple", "teal"]
    font_color = dark_colors
    codes = "123456789abcdefghjkmnpqrstuvwxyzABCDEFGHJKMNPQRSTUVWXYZ"
    background = (random.randrange(150, 255), random.randrange(150, 255), random.randrange(150, 255))
    line_color = (random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255))
    sample_file = os.path.join(os.path.dirname(__file__), './LucidaSansDemiOblique.ttf')
    font = ImageFont.truetype(sample_file, font_size)
    image = Image.new('RGB', (width, height), background)
    # draw = ImageDraw.Draw(image)
    code = ''.join(random.sample(codes, chars_num))
    draw = ImageDraw.Draw(image)
    for i in range(random.randrange(5, 10)):
        xy = (random.randrange(0, width), random.randrange(0, height),
              random.randrange(0, width), random.randrange(0, height))
        draw.line(xy, fill=line_color, width=1)
    x = 2
    for i in code:
        y = random.randrange(0, 10)
        draw.text((x, y), i, font=font, fill=random.choice(font_color))
        x += font_size - 2
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return image, code
    # buf = StringIO()
    # 
    # image.save(buf, 'jpeg')
    # buf.seek(0)
    # return code, buf

def gen_custom(gen_dir, total_size, chars_num):
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    
    for i in xrange(total_size):
        image, label = generate_captcha(width=300, height=100,font_size=80, chars_num=chars_num)
        file_path = os.path.join(gen_dir, label+'_num'+str(i)+'.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, 'jpeg')


def gen(gen_dir, total_size, chars_num):
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,font_sizes=[40])
    # must be subset of config.CHAR_SETS
    # char_sets = 'ABCDEFGHIJKLMNPQRSTUVWXYZ'
    char_sets = config.CHAR_SETS
    for i in xrange(total_size):
        label = ''.join(random.sample(char_sets, chars_num))
        image.write(label, os.path.join(gen_dir, label+'_num'+str(i)+'.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--test_dir',
            type=str,
            default='./data/test_data',
            help='Directory testing to generate captcha data files'
    )
    parser.add_argument(
            '--train_dir',
            type=str,
            default='./data/train_data',
            help='Directory training to generate captcha data files'
    )
    parser.add_argument(
            '--valid_dir',
            type=str,
            default='./data/valid_data',
            help='Directory validation to generate captcha data files'
    )
    FLAGS, unparsed = parser.parse_known_args()
    print('>> generate custom %d captchas in %s' % (TEST_SIZE, FLAGS.test_dir))
    gen_custom(FLAGS.test_dir, TEST_SIZE, CHARS_NUM)
    print('>> generate custom %d captchas in %s' % (TRAIN_SIZE, FLAGS.train_dir))
    gen_custom(FLAGS.train_dir, TRAIN_SIZE, CHARS_NUM)
    print('>> generate custom %d captchas in %s' % (VALID_SIZE, FLAGS.valid_dir))
    gen_custom(FLAGS.valid_dir, VALID_SIZE, CHARS_NUM)
    print('>> generate %d captchas in %s' % (TEST_SIZE, FLAGS.test_dir))
    gen(FLAGS.test_dir, TEST_SIZE, CHARS_NUM)
    print ('>> generate %d captchas in %s' % (TRAIN_SIZE, FLAGS.train_dir))
    gen(FLAGS.train_dir, TRAIN_SIZE, CHARS_NUM)
    print ('>> generate %d captchas in %s' % (VALID_SIZE, FLAGS.valid_dir))
    gen(FLAGS.valid_dir, VALID_SIZE, CHARS_NUM)
    print ('>> generate Done!')
    

    # gen_custom('./data/train_data', 1, 4)
