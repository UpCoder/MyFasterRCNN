# -*- coding=utf-8 -*-
from util.dl_components import do_conv, do_maxpooling, do_reshape, do_fc
import tensorflow as tf
from anchor_target import anchor_target_layer
from proposal_layer_tf import proposal_layer
from proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
from lib.roi_pooling_layer import roi_pooling_op
from roi_pooling.roi_pooling_ops import roi_pooling

_feat_stride = [16,]
anchor_scales = [8, 16, 32]
n_classes = 21

def softmax(input, name):
    input_shape = tf.shape(input)
    if name == 'rpn_cls_prob':
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
    else:
        return tf.nn.softmax(input, name=name)


def vgg16(input_image_tensor, input_gt_box_tensor, input_im_info_tensor, is_training=True):
    # extract feature map
    conv1_1 = do_conv(input_image_tensor, 'conv1_1', [3, 3], 64, [1, 1], activation_method=tf.nn.relu, trainable=False)
    conv1_2 = do_conv(conv1_1, 'conv1_2', [3, 3], 64, [1, 1], activation_method=tf.nn.relu, trainable=False)
    pool1 = do_maxpooling(conv1_2, padding='VALID', layer_name='pool1', kernel_size=[2, 2], stride_size=[2, 2])
    conv2_1 = do_conv(pool1, 'conv2_1', [3, 3], 128, [1, 1], activation_method=tf.nn.relu, trainable=False)
    conv2_2 = do_conv(conv2_1, 'conv2_2', [3, 3], 128, [1, 1], activation_method=tf.nn.relu, trainable=False)
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
    rpn_cls_score = do_conv(rpn_conv, layer_name='rpn_cls_score', kernel_size=[1, 1], filter_size=18,
                            stride_size=[1, 1],
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
    rpn_data = []
    rpn_data.append(rpn_labels)
    rpn_data.append(rpn_bbox_targets)
    rpn_data.append(rpn_bbox_inside_weights)
    rpn_data.append(rpn_bbox_outside_weights)

    rpn_bbox_pred = do_conv(rpn_conv, layer_name='rpn_bbox_pred', kernel_size=[1, 1], filter_size=36,
                            stride_size=[1, 1], activation_method=None, padding='VALID')
    # reshape to N, W/2*18, H, 2,why? I guess because we need to softmax which need normalization the probability.
    rpn_cls_score_reshape = do_reshape(rpn_cls_score, 2, name='rpn_cls_score_reshape')
    rpn_cls_prob = softmax(rpn_cls_score_reshape, name='rpn_cls_prob')
    # reshape to N, W, H, 18
    rpn_cls_prob_reshape = do_reshape(rpn_cls_prob, len(anchor_scales)*3*2, name='rpn_cls_prob_reshape')
    # 使用非最大抑制来选择bbox，也就是proposal
    rpn_rois = tf.reshape(tf.py_func(proposal_layer,
                                     [rpn_cls_prob_reshape, rpn_bbox_pred, input_im_info_tensor, 'TRAIN', _feat_stride,
                                      anchor_scales], [tf.float32]), [-1, 5], name='rpn_rois')
    # proposal 分为fg和bg以及两者都不是
    rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer_py,
                                                                                       [rpn_rois, input_gt_box_tensor,
                                                                                        n_classes],
                                                                                       [tf.float32, tf.float32,
                                                                                        tf.float32, tf.float32,
                                                                                        tf.float32])
    rois = tf.reshape(rois, [-1, 5], name='rois')
    labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
    bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
    bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
    bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')
    roi_data = []
    roi_data.append(rois)
    roi_data.append(labels)
    roi_data.append(bbox_targets)
    roi_data.append(bbox_inside_weights)
    roi_data.append(bbox_outside_weights)

    # ======R-CNN===========
    print conv5_3, rois
    print conv5_3.get_shape(), rois.get_shape(), tf.cast(tf.multiply(rois, 1.0/16.0)[:, 1:], tf.int32).get_shape()
    # roi_pool = roi_pooling_op.roi_pool(conv5_3, tf.cast(rois, tf.float32), 7, 7, 1.0/16, name='roi_pool')[0]
    # roi_pool = roi_pooling(conv5_3, tf.cast(tf.multiply(rois, 1.0/16.0)[:, 1:], tf.int32), 7, 7)
    roi_pool = tf.image.crop_and_resize(conv5_3, tf.multiply(rois, 1.0 / 16.0)[:, 1:],
                                        tf.cast(rois[:, 0], tf.int32), [7, 7])
    # roi_pool.set_shape([None, 7, 7, 512])
    print roi_pool.get_shape()

    fc1 = do_fc(roi_pool, 'fc1', output_node=4096)
    if is_training:
        fc1 = tf.nn.dropout(fc1, 0.5)
    fc2 = do_fc(fc1, 'fc2', output_node=4096)
    if is_training:
        fc2 = tf.nn.dropout(fc2, 0.5)
    cls_score = do_fc(fc2, 'cls_score', output_node=n_classes, relu=False)
    # 预测每个proposal的类别
    cls_prob = tf.nn.softmax(cls_score, name='cls_prob')
    # 预测每个proposal的坐标
    bbox_pred = do_fc(fc2, 'bbox_pred', n_classes*4, relu=False)
    return rpn_data, roi_data, rpn_cls_score_reshape, rpn_bbox_pred, conv5_3, cls_score, cls_prob, bbox_pred
