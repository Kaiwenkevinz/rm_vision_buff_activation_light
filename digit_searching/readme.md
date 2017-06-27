## digit_searching.py

Requirements: numpy, cv2, moviepy

Description: It will search and identify digits from image or video input.

How to use: 
Save the vedio under the same folder and run the .py file. If images are used, they should be passed in `process_image()` with `debug=True`
The digits can be accessed by `my_digits.result_digits`.

How it works:

[//]: # (Image References)

[image1]: ./output_images/masked.jpg "Masked Image"
[image2]: ./output_images/rgb_select.jpg "Rgb Select"
[image3]: ./output_images/gray_select.jpg "gray Select"
[image4]: ./output_images/blob_detection.jpg "Blob Detection"
[image5]: ./output_images/five_blobs.jpg "Five Blobs"
[image6]: ./output_images/digits_region.jpg "Digits Region"
[image7]: ./output_images/binary_digits.jpg "Binary Digits"
[image8]: ./output_images/rotated.jpg "Rotated"
[image9]: ./output_images/result.jpg "result"

1. Only the central part will be analysed.

![alt text][image1]

2. Select red region to find digits

![alt text][image2]

3. Select gray region to find digits more precisely

![alt text][image3]

4. The red region is used in blob detection as teh blobs are much easier to detect in this image.

![alt text][image4]

5. Select five blobs which are cloest in y

![alt text][image5]

6. Use these five blobs to find the digit region

![alt text][image6]

7. It is easier to do the identification in binary image

![alt text][image7]

8. Rotate the image a little bit to make the vertical line more vertical

![alt text][image8]

9. The final result is like this

![alt text][image9]

10. Print digits on images
