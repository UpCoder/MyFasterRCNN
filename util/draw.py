from PIL import Image, ImageDraw
import numpy as np
from keras.preprocessing.image import array_to_img


def show_image(image_arr):
    img = Image.fromarray(image_arr)
    img.show()


def draw_rects_image(image_arr, rects):
    img = array_to_img(np.squeeze(image_arr).transpose([1, 0, 2]))
    img.show('original')
    draw = ImageDraw.Draw(img)
    for rect in rects:
        draw.rectangle(rect)
    img.show()


if __name__ == '__main__':
    img = Image.open(
        '/home/give/Game/OCR/Papers-code/Faster-RCNN_TF-master/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg')
    draw_rects_image(np.array(img), [[0, 0, 100, 100], [100, 100, 150, 150]])