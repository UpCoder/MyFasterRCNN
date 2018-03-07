# -*- coding=utf-8 -*-
from util.dl_components import do_conv, do_maxpooling, do_reshape, _modified_smooth_l1
import tensorflow as tf
from anchor_target import anchor_target_layer
import sys
import argparse

_feat_stride = [16,]
anchor_scales = [8, 16, 32]

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


def vgg16(input_image_tensor, input_gt_box_tensor, input_im_info_tensor):
    # extract feature map
    conv1_1 = do_conv(input_image_tensor, 'conv1_1', [3, 3], 64, [1, 1], activation_method=tf.nn.relu)
    conv1_2 = do_conv(conv1_1, 'conv1_2', [3, 3], 64, [1, 1], activation_method=tf.nn.relu)
    pool1 = do_maxpooling(conv1_2, padding='VALID', layer_name='pool1', kernel_size=[2, 2], stride_size=[2, 2])
    conv2_1 = do_conv(pool1, 'conv2_1', [3, 3], 128, [1, 1], activation_method=tf.nn.relu)
    conv2_2 = do_conv(conv2_1, 'conv2_2', [3, 3], 128, [1, 1], activation_method=tf.nn.relu)
    pool2 = do_maxpooling(conv2_2, layer_name='pool2', padding='VALID', kernel_size=[2, 2], stride_size=[2, 2])
    conv3_1 = do_conv(pool2, 'conv3_1', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    conv3_2 = do_conv(conv3_1, 'conv3_2', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    conv3_3 = do_conv(conv3_2, 'conv3_3', kernel_size=[3, 3], filter_size=256, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    pool3 = do_maxpooling(conv3_3, layer_name='pool3', padding='VALID', kernel_size=[2, 2], stride_size=[2, 2])
    conv4_1 = do_conv(pool3, layer_name='conv4_1', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    conv4_2 = do_conv(conv4_1, layer_name='conv4_2', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    conv4_3 = do_conv(conv4_2, layer_name='conv4_3', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    pool4 = do_maxpooling(conv4_3, layer_name='pool4', padding='VALID', kernel_size=[2, 2], stride_size=[2, 2])
    conv5_1 = do_conv(pool4, layer_name='conv5_1', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    conv5_2 = do_conv(conv5_1, layer_name='conv5_2', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      activation_method=tf.nn.relu)
    conv5_3 = do_conv(conv5_2, layer_name='conv5_3', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                      activation_method=tf.nn.relu)

    # Region Proposal Network
    rpn_conv = do_conv(conv5_3, layer_name='rpn_conv', kernel_size=[3, 3], filter_size=512, stride_size=[1, 1],
                       activation_method=tf.nn.relu)
    rpn_cls_score = do_conv(rpn_conv, layer_name='rpn_cls_score', kernel_size=[1, 1], filter_size=18, stride_size=[1, 1],
                        activation_method=None, padding='VALID')
    # rpn_labels表示提取所有的bbox的label
    # rpn_bbox_targets表示的是提取的所有bbox的中心坐标和长宽和gt的差值
    # rpn_bbox_inside_weights 表示的是系数，label不为1的为0， 用于计算Smooth L1 loss
    # rpn_bbox_outside_weights 表示的是系数，用于计算Smooth L1 loss时的正则化
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
               [rpn_cls_score, input_gt_box_tensor, input_im_info_tensor, input_image_tensor, _feat_stride, anchor_scales],
               [tf.float32, tf.float32, tf.float32, tf.float32])
    rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
    rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
    rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
    rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

    rpn_bbox_pred = do_conv(rpn_conv, layer_name='rpn_bbox_pred', kernel_size=[1, 1], filter_size=36,
                            stride_size=[1, 1], activation_method=None, padding='VALID')
    rpn_cls_score_reshape = do_reshape(rpn_cls_score, 2, name='rpn_cls_score_reshape')

    # RPN
    # classification loss
    rpn_cls_score = tf.reshape(rpn_cls_score_reshape, [-1, 2])
    rpn_label = tf.reshape(rpn_labels, [-1])
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
    rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])
    rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

    # bounding box regression L1 loss
    rpn_bbox_pred = rpn_bbox_pred
    rpn_bbox_targets = tf.transpose(rpn_bbox_targets, [0, 2, 3, 1])
    rpn_bbox_inside_weights = tf.transpose(rpn_bbox_inside_weights, [0, 2, 3, 1])
    rpn_bbox_outside_weights = tf.transpose(rpn_bbox_outside_weights, [0, 2, 3, 1])

    rpn_smooth_l1 = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                        rpn_bbox_outside_weights)
    rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3]))

    loss = rpn_cross_entropy + rpn_loss_box

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
        print bbox_means, bbox_stds
        data_layer = get_data_layer(roidb, imdb.num_classes)
        blobs = data_layer.forward()
        print blobs.keys()
        for key in blobs.keys():
            print key, np.shape(blobs[key])
            if key != 'data':
                print blobs[key]
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        rpn_labels_value, rpn_bbox_targets_value, rpn_bbox_inside_weights_value, rpn_bbox_outside_weights_value = sess.run(
            [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights], feed_dict={
                input_image_tensor: blobs['data'],
                input_gt_box_tensor: blobs['gt_boxes'],
                input_im_info_tensor: blobs['im_info']
            })
        print np.shape(rpn_labels_value), np.shape(rpn_bbox_targets_value), np.shape(
            rpn_bbox_inside_weights_value), np.shape(rpn_bbox_outside_weights_value)
        conv5_3_value, loss_value = sess.run([conv5_3, loss], feed_dict={
            input_image_tensor: blobs['data'],
            input_gt_box_tensor: blobs['gt_boxes'],
            input_im_info_tensor: blobs['im_info']
        })
        print np.shape(conv5_3_value)
        print 'loss value is ', loss_value

if __name__ == '__main__':
    data_tensor = tf.placeholder(tf.float32, [None, None, None, 3], name='x-input')
    input_gt_box_tensor = tf.placeholder(tf.float32, [None, 5], name='gt_box_input')
    input_im_info_tensor = tf.placeholder(tf.float32, [None, 3], name='im_info_input')
    vgg16(data_tensor, input_gt_box_tensor, input_im_info_tensor)

