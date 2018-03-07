# -*- coding=utf-8 -*-
import numpy as np


def bbox_overlaps(boxes, query_boxes):
    '''

    :param boxes: (N, 4) ndarray of float
    :param query_boxes: (K, 4) ndarray of float
    :return: (N, K) ndarray of overlap between boxes and query_boxes
    '''
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    print np.shape(boxes)
    print np.shape(query_boxes)
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            # 也就是说最小的左下方的横坐标减去最大的右上方的横坐标代表的就是IoU部分的宽度
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
