import tensorflow as tf
from network import vgg16
from lib.datasets.factory import get_imdb
from lib.fast_rcnn.train import get_training_roidb, rdl_roidb, get_data_layer, filter_roidb
import numpy as np
from util.draw import draw_rects_image


def test(input_image_tensor, input_gt_box_tensor, input_im_info_tensor, model_path, ):
    print 'ok'


if __name__ == '__main__':
    data_tensor = tf.placeholder(tf.float32, [None, None, None, 3], name='x-input')
    input_gt_box_tensor = tf.placeholder(tf.float32, [None, 5], name='gt_box_input')
    input_im_info_tensor = tf.placeholder(tf.float32, [None, 3], name='im_info_input')
    rpn_data, roi_data, rpn_cls_score_reshape, rpn_bbox_pred, conv5_3= vgg16(
        data_tensor, input_gt_box_tensor, input_im_info_tensor)
    model_path = '/home/give/PycharmProjects/MyFasterRCNN/parameters'
    from lib.fast_rcnn.config import cfg_from_file, cfg_from_list
    from RPN.train import parse_args
    args = parse_args()
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
    print np.shape(blobs['data'])
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # restore model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print blobs.keys()
        print np.shape(blobs['gt_boxes'])
        rpn_rois_values = sess.run(roi_data[0], feed_dict={
            data_tensor: blobs['data'],
            input_im_info_tensor: blobs['im_info'],
            input_gt_box_tensor: blobs['gt_boxes']
        })
        print np.shape(rpn_rois_values)
        draw_rects_image(blobs['data'], rpn_rois_values[:, 0:4])