import tensorflow as tf
from lib.fast_rcnn.config import cfg
import os
from util.dl_components import _modified_smooth_l1
import argparse
import sys
from network import vgg16


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def train(rpn_data, roi_data,rpn_cls_score_reshape, rpn_bbox_pred, feature_map, cls_score, bbox_pred, input_image_tensor, input_gt_box_tensor, input_im_info_tensor):
    output_dir = '/home/give/PycharmProjects/MyFasterRCNN/parameters'
    saver = tf.train.Saver(max_to_keep=5)
    # RPN
    # classification loss(fg bg)
    rpn_cls_score = tf.reshape(rpn_cls_score_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_data[0], [-1])
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])
    rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    # bounding box regression L1 loss
    rpn_bbox_pred = rpn_bbox_pred
    rpn_bbox_targets = tf.transpose(rpn_data[1], [0, 2, 3, 1])
    rpn_bbox_inside_weights = tf.transpose(rpn_data[2], [0, 2, 3, 1])
    rpn_bbox_outside_weights = tf.transpose(rpn_data[3], [0, 2, 3, 1])

    rpn_smooth_l1 = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                        rpn_bbox_outside_weights)
    rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

    # classification loss
    label = tf.reshape(roi_data[1], [-1])
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

    smooth_l1 = _modified_smooth_l1(1.0, bbox_pred, roi_data[2], roi_data[3], roi_data[4])
    loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step,
                                    cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
    momentum = cfg.TRAIN.MOMENTUM
    train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        args = parse_args()
        from lib.datasets.factory import get_imdb
        from lib.fast_rcnn.train import get_training_roidb, filter_roidb
        from lib.fast_rcnn.train import get_data_layer
        import lib.roi_data_layer.roidb as rdl_roidb
        import numpy as np
        from lib.fast_rcnn.config import cfg_from_file, cfg_from_list
        if args.cfg_file is not None:
            cfg_from_file(args.cfg_file)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs)
        imdb = get_imdb('voc_2007_trainval')
        roidb = get_training_roidb(imdb)
        roidb = filter_roidb(roidb)
        bbox_means, bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        data_layer = get_data_layer(roidb, imdb.num_classes)
        blobs = data_layer.forward()
        for key in blobs.keys():
            print key, np.shape(blobs[key])
            if key != 'data':
                print blobs[key]
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for iter_index in range(args.max_iters):
            _, rpn_labels_value, rpn_bbox_targets_value, rpn_bbox_inside_weights_value, rpn_bbox_outside_weights_value = sess.run(
                [train_op, rpn_data[0], rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights], feed_dict={
                    input_image_tensor: blobs['data'],
                    input_gt_box_tensor: blobs['gt_boxes'],
                    input_im_info_tensor: blobs['im_info']
                })
            conv5_3_value, loss_value, rpn_cross_entropy_value, rpn_loss_box_value = sess.run([feature_map, loss, rpn_cross_entropy, rpn_loss_box], feed_dict={
                input_image_tensor: blobs['data'],
                input_gt_box_tensor: blobs['gt_boxes'],
                input_im_info_tensor: blobs['im_info']
            })
            if iter_index % 1000 == 0:
                print 'iter: %d / %d' % (iter_index, args.max_iters)
                print 'total loss: %.4f, rpn cross entropy: %.4f,  rpn_loss_box: %.4f' % (loss_value, rpn_cross_entropy_value, rpn_loss_box_value)
                if iter_index == 0:
                    continue
                infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                         if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
                filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                            '_iter_{:d}'.format(iter_index + 1) + '.ckpt')
                filename = os.path.join(output_dir, filename)

                saver.save(sess, filename)

if __name__ == '__main__':
    data_tensor = tf.placeholder(tf.float32, [None, None, None, 3], name='x-input')
    input_gt_box_tensor = tf.placeholder(tf.float32, [None, 5], name='gt_box_input')
    input_im_info_tensor = tf.placeholder(tf.float32, [None, 3], name='im_info_input')
    rpn_data, roi_data, rpn_cls_score_reshape, rpn_bbox_pred, conv5_3, cls_score, bbox_pred = vgg16(data_tensor,
                                                                                                   input_gt_box_tensor,
                                                                                                   input_im_info_tensor)
    train(rpn_data, roi_data, rpn_cls_score_reshape, rpn_bbox_pred, conv5_3, cls_score, bbox_pred, data_tensor,
          input_gt_box_tensor,
          input_im_info_tensor)