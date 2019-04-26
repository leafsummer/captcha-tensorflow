from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import os
import sys

import tensorflow as tf
import captcha_model as captcha

FLAGS = None

def run_train():
    """Train CAPTCHA for a number of steps."""

    with tf.Graph().as_default():
        images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)

        logits = captcha.inference(images, keep_prob=0.5)

        loss = captcha.loss(logits, labels)

        train_op = captcha.training(loss)

        saver = tf.train.Saver(tf.global_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                                             tf.local_variables_initializer())

        sess = tf.Session()

        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        checkpoint_path = os.path.join(FLAGS.model_dir, 'checkpoint')
        print("checkpoint_path: %s" % checkpoint_path)
        try:
            step = 0
            while not coord.should_stop():
                start_time = time.time()
                _, loss_value = sess.run([train_op, loss])
                duration = time.time() - start_time
                if step % 10 == 0:
                    print('>> Step %d run_train: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                                            duration))
                if step % 100 == 0:
                    print('>> %s Saving in %s' % (datetime.now(), checkpoint_path))
                    saver.save(sess, checkpoint_path, global_step=step)
                step += 1
        except Exception as e:
            print('>> %s Saving in %s' % (datetime.now(), checkpoint_path))
            saver.save(sess, checkpoint_path, global_step=step)
            coord.request_stop(e)
        finally:
            coord.request_stop()
        coord.join(threads)
        sess.close()


def main(_):
    if tf.gfile.Exists(FLAGS.model_dir):
        tf.gfile.DeleteRecursively(FLAGS.model_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)
    run_train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--batch_size',
            type=int,
            default=128,
            help='Batch size.'
    )
    parser.add_argument(
            '--model_dir',
            type=str,
            default='./models',
            help='Directory where to write models.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
