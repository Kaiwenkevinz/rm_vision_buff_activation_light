import serial

# These are predefined values for the communication protocol:
MSG_START = 0xff
MSG_STOP = 0xfe
MSG_CONFIRM = 0xff
MSG_AXIS = 0xfe

class Communication:
    def __init__(self, port = '/dev/ttyTHS2', bitrate = 115200):
        self.ser = serial.Serial(port, bitrate, timeout = 0)

    def send(self, data):
        msg = bytearray(data)
        self.ser.write(msg)

    def receive(self):
        msg = self.ser.read(1)
        if len(msg) != 1:
            return None
        else:
            return ord(msg)

    def close(self):
        self.ser.close()

    def send_axis(self, val):
        """
        val: [-128, 127],  which is encoded as [0, 255]
        """
        assert val >= -128 and val <= 127
        self.send([val + 128])
