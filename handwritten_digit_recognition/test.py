import numpy as np
import cv2
from classifier import HandwrittenDigitClassifier

handwritten_digit_classifier = HandwrittenDigitClassifier()
img = cv2.imread('test/1.png')
prob = handwritten_digit_classifier.predict(img)
prediction = np.argmax(prob)
print prediction
