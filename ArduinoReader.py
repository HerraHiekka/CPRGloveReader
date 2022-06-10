import numpy as np
import pandas as pd

from serial import Serial
from serial.threaded import LineReader, Protocol, ReaderThread

import os
import time
import re
import threading

from helpers import is_float

# Super init:
# (port=None, baudrate=9600, bytesize=EIGHTBITS, parity=PARITY_NONE, stopbits=STOPBITS_ONE, timeout=None,
# xonxoff=False, rtscts=False, write_timeout=None, dsrdtr=False, inter_byte_timeout=None, exclusive=None)
# Optional inclusions:
# save: string - Save the current session to provided path

class ArduinoReader:

    def __init__(self, cols, save=None, **kwargs):
        self.serial = Serial(**kwargs)
        self.main_thread = ReaderThread(self.serial, ArduinoReader.SerialReader)
        self.data = []
        self.cols = cols
        self.save = save
        pass

    def start(self):
        self.main_thread.start()

    def get_data(self):
        if self.main_thread.is_alive():
            self.data = []
            new_data = self.main_thread.protocol.received_lines[5:]
            new_data = [re.split(",", s) for s in new_data]
            new_data = [a for a in new_data if len(a) == 10 and is_float(a)]
            new_data = np.array(new_data).astype("float")
            self.data.extend(new_data)
            self.data = pd.DataFrame(self.data, columns=self.cols)
        return self.data

    def stop(self):
        data = self.get_data()
        self.main_thread.stop()
        if self.save is not None:
            data.to_csv(f'./recordings/{self.save}/data.csv')

    class SerialReader(LineReader):
        def __init__(self):
            super(ArduinoReader.SerialReader, self).__init__()
            self.received_lines = []

        def handle_line(self, data):
            self.received_lines.append(data)


if __name__ == '__main__':
    reader_args = {
        "port": "/dev/ttyACM0",
        "baudrate": 9600,
        "timeout": 0
    }
    reader = ArduinoReader(cols=["time", "press1", "press2", "press3", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"],
                           save=False,
                           **reader_args)
    print('Starting reader')
    reader.start()
    print('waiting for data collection')
    time.sleep(3)
    print('printing data:')
    reader.get_data()
    time.sleep(3)
    reader.get_data()
    print(reader.data)
    reader.stop()

    T = reader.data.time.diff().mean()
    print(T)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(reader.data.acc1)
    plt.plot(reader.data.acc2)
    plt.plot(reader.data.acc3)
    plt.show()

    from scipy.fftpack import fft, fftfreq
    from scipy.signal import find_peaks

    N = len(reader.data.acc1)
    yf = fft(reader.data.acc1.to_numpy().reshape(-1,))
    yf = 2.0 / N * np.abs(yf[0:N // 2])
    xf = fftfreq(N, T/1000)[:N // 2]
    peaks, properties = find_peaks(yf)
    i = np.argmax(yf[peaks])
    peak = peaks[i]

    plt.figure()
    plt.plot(xf, yf)
    plt.scatter(xf[peak], yf[peak], marker='x', color='r')
    plt.show()

