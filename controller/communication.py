import serial

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
            print hex(ord(msg))
            return msg

    def start_received(self):
        msg = self.receive()
        return msg != None and ord(msg) == 0xff

    def stop_received(self):
        msg = self.receive()
        return msg != None and ord(msg) == 0xfe

    def close(self):
        self.ser.close()

    def send_axis(self, val):
        """
        val: [-128, 127],  which is encoded as [0, 255]
        """
        assert val >= -128 and val <= 127
        self.send([val + 128])

if __name__ == '__main__':
    communication = Communication()

    while True:
        msg = communication.receive()
        if msg is not None and ord(msg) == 0xff:
            communication.send([0xff])
            communication.send([128])
            communication.send([128])
        
        msg = communication.receive()
        if msg is not None:
            print msg
    
    communication.close()
