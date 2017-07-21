import serial

class Communication:
    def __init__(self, port = '/dev/ttyTHS2', bitrate = 115200):
        self.ser = serial.Serial(port, bitrate, timeout = 1)

    def send(self, data):
        msg = bytearray(data)
        self.ser.write(msg)

    def close(self):
        self.ser.close()
