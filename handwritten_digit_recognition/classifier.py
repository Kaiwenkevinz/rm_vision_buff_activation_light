import cv2, caffe, os
import numpy as np
from scipy.misc import imresize

image_h = 28
image_w = 28

file_dir = os.path.dirname(os.path.abspath(__file__))
model_def = file_dir + '/lenet_train_test_deploy.prototxt'
model_weights = file_dir + '/weights/lenet.caffemodel'

class HandwrittenDigitClassifier:
    def __init__(self, mode = 'GPU'):
        if mode == 'GPU':
            caffe.set_device(0)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})

    def predict(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.0
        img = imresize(img, [image_h, image_w])
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        output = self.net.forward()
        prob = output['prob'][0] # the prob for the first image (the only image)
        return prob
