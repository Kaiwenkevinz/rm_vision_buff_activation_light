import cv2
from led import LedDisplaysRecognizer

if __name__ == '__main__':
    cam = cv2.VideoCapture('../test0.mpeg')
    led_displays_recognizer = LedDisplaysRecognizer()

    while True:
        ret, frame = cam.read()
        assert ret == True

        # Select region of interest (roi)
        h, w, _ = frame.shape
        rh = int(round(h * 0.2))
        rw = int(round(w * 0.2))
        roi_y0 = rh
        roi_y1 = h - rh
        roi_x0 = rw
        roi_x1 = w - rw
        cv2.rectangle(frame, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 2)

        result = led_displays_recognizer.process(frame[roi_y0:roi_y1, roi_x0:roi_x1, :])
        if result != None:
            digits, x0, x1, y0, y1 = result
            x0 += rw
            y0 += rh
            cv2.putText(frame, digits, (x0, y0), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))
        else:
            cv2.putText(frame, 'Not Found', (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (128, 255, 0))


        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
