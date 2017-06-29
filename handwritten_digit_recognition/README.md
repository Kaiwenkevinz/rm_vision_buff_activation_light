# Handwritten Digit Recognition

This is a handwritten digit classifier using Convolutional Neural Net
implemented in Caffe.

We trained two layered conv net on MNIST dataset with data
augmentation to prevent overfitting, the original dataset has been randomly
translated, rotated, added black edges and random noise, so that the training
data is more close to what a camera can see in reality.

## Usage

We have already include the weights file so you don't have to train from the
scratch, but you can do that by running:
```
./init.sh
./train-lenet.sh
```

The classifier has been written into a module, you can import it by doing the
following in you python code:
```Python
import numpy as np
from classifier import HandwrittenDigitClassifier
handwritten_digit_classifier = HandwrittenDigitClassifier()
img = # something
prob = handwritten_digit_classifier.predict(img)
prediction = np.argmax(prob)
```
