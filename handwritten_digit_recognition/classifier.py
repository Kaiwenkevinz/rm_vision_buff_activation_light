import cv2
import caffe
import numpy as np
from scipy.misc import imresize

image_h = 28
image_w = 28

model_def = 'lenet_train_test_deploy.prototxt'
model_weights = 'weights/lenet.caffemodel'

class HandwrittenDigitClassifier:
    def __init__(self, mode = 'GPU'):
        if mode == 'GPU':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = imresize(img, [image_h, image_w])
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        output = self.net.forward()
        prob = output['prob'][0] # the prob for the first image (the only image)
        return prob
