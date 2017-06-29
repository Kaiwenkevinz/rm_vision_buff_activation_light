
# rm_vision_buff_activation

## number identification with CNN done
cnn_training.py

requirement:
tensorflow
numpy

description:
It builts a convolutional neurtal network with 2 convolutional layers and 2 fully connected layers.
Then it trains CNN with 2000 steps. More training is allowed as it does not show overfitting. But that might be unnecessary because the structure is pretty simple.

number_detection.py

requirment:
tensorflow
numpy
glob
cv2

description:
It will use the model created above to identify numbers from pictures in *test_images*.
Save pictures (.png) to test_images folder and all of them will be identified. If .jpg images are used, just change *.png to .jpg. The `process_image()` crops the boundry by 10 pixels to prevent any black boundries being passed to model.

