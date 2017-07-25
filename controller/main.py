from communication import *
from rune_recognition import RuneRecognition
import time

class MainController:
    def __init__(self):
        self.communication = Communication()
        self.rune_recognition = RuneRecognition()
        self.loop()

    def loop(self):
        """
        Control start/stop and communication with the driver:
        """
        while True:
            msg = self.communication.receive()
            if msg == MSG_START:
                self.communication.send([MSG_CONFIRM]) # confirmation respond
                while True:
                    axis = self.process() # process a frame
                    if axis != None:
                        x_axis, y_axis = axis
                        self.communication.send([MSG_AXIS]) # trigger driver state machine to receive axis
                        self.communication.send_axis(x_axis) # send x axis
                        self.communication.send_axis(y_axis) # send y axis
                    if self.communication.receive() == MSG_STOP:
                        # Suspend the program
                        break

    def process(self):
        return self.rune_recognition.process()
        # x_axis = 120
        # y_axis = -90
        # return x_axis, y_axis

    def close(self):
        self.communication.close()
        self.rune_recognition.close()

if __name__ == '__main__':
    main_controller = MainController()
    main_controller.close()
