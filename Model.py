import numpy as np
import pandas as pd

from scipy.integrate import cumtrapz
from scipy.signal import filtfilt, butter, find_peaks
from scipy.fftpack import fft,  fftfreq
from sklearn.decomposition import PCA

from enums import State

# Computes the depth, frequency and lean estimates, returns them in addition to the appropriate state
class Model:

    def __init__(self, freq_bounds=np.array([90,140]), depth_bounds=np.array([5.0, 6.0]), offset=4, eps=1.0, calibration=False):
        self.offset = offset
        self.eps = eps # Pressure sensor threshold
        self.f_bounds = freq_bounds
        self.d_bounds = depth_bounds
        self.calibration = calibration

    def estimate(self, data):
        # Sampling period, converted to seconds from ms.
        self.T = data.time.diff().mean() / 1000.

        # Take only the last 3 seconds of data, as the previous compressions should be found there
        n = int(np.ceil(3. / self.T)) # 3s divided by sampling period to obtain the number of samples we go through
        data = data[-n:]
        # Resetting the index for ease of use later
        data.reset_index(inplace=True)

        # PCA to get the acceleration to the direction of compressions
        self.acc = self._find_dir(data)

        # Get the points where each compression is started
        self.transitions = self._find_rest_states(data)

        # If there are no noticable peaks (or if there are far too many), assume compressions are not happening
        if len(self.transitions) < self.offset\
                or len(self.transitions) > self.offset * 3:
            return State.INTERRUPTION

        if not np.any(data.press1[self.transitions[-3:]] > self.eps):
            #print(f'Last 3 compression depths were {data.press1[self.transitions[-3:]]}, allowed bound is {self.eps}')
            return State.LEAN

        # Next, we obtain the frequency based on the rest-state transitions and a second estimate based on the frequency
        self.transition_f = self._get_transition_frequency(data)
        self.frequency = self._get_fft_frequency(self.acc)

        # Check if the frequency is lower or higher than provided bounds
        low_freq = high_freq = False
        if self.frequency > self.f_bounds[1]:
            return State.HIGH_RATE
        elif self.frequency < self.f_bounds[0]:
            #print(f"Rate: {self.frequency}, bound: {self.f_bounds[0]}")
            return State.LOW_RATE

        # Finally, we obtain depth by first applying the double integration and then estimating the depth
        vels, ds = self._transform(self.acc)

        # And then finding the peak-to-peak depth
        self.depth = self._get_depth_from_diffs(ds)

        # Check That the depth is within appropriate bounds
        if np.all(self.depth < self.d_bounds[0]):
            return State.LOW_DEPTH
        elif np.all(self.depth > self.d_bounds[1]):
            return State.HIGH_DEPTH

        return State.OK

    def _get_fft_frequency(self, data):
        if self.transitions is None:
            raise Exception("Model: Transitions not set for frequency")
        # This can be done based on acceleration

        # Simply finding peaks from frequency spectrum. Highest peak is assumed to be the
        # correct one. If no peak is found, simple 0 is written. Statistical properties
        # of a peak could be considered for a reliable peak detection in this case.
        # Wavelet based peak finding could also be considered.
        data = data[self.transitions[-(self.offset)]:self.transitions[-1]]
        N = len(data)
        yf = fft(data.reshape(-1,))
        yf = 2.0 / N * np.abs(yf[0:N // 2])
        xf = fftfreq(N, self.T)[:N // 2]
        peaks, properties = find_peaks(yf, prominence=1)
        if len(peaks) == 0:
            return 0.0
        else:
            peak = peaks[0] # Max peak seems to sometimes be a harmonic, while the first peak is never the trendline
            peak_x = xf[peak]

        return peak_x * 60 # Returning the frequency as compressions per minute

    def _get_transition_frequency(self, data):
        # Get timestamps for the transitions
        times = data.time[self.transitions[-(self.offset)]:]
        # Calculate the mean time difference
        T = times.diff().mean()
        # Output the frequency in compressions per minute
        return 60./T

    def _get_depth_from_diffs(self, data):
        if self.transitions is None:
            raise Exception("Model: Transitions not set for depth")
        transitions = self.transitions[-self.offset:]

        # Get the depth for each compression separately, then average
        depths = []
        for start, stop in zip(transitions[:-1]-transitions[0], transitions[1:]-transitions[0]):
            d = np.max(data[start:stop]) - np.min(data[start:stop])
            depths.append(d)
        depth = np.mean(depths)

        return np.array(depths) * 100 # Return the depth in cm (original is in m)

    def _transform(self, data):
        # We apply the method only to the three latest compressions, i.e. only using the last three crossings from mask
        transitions = self.transitions[-self.offset:]
        data = data[self.transitions[-self.offset]:]

        # Applying only the ending points of the masks, as
        # this appeared to be the best for the depth estimates
        vels = np.zeros(data.shape)
        ds = np.zeros(data.shape)
        # Apply the integration to each compression separately
        for start, stop in zip(transitions[:-1]-transitions[0], transitions[1:]-transitions[0]):
            vels[start:stop] = self._integrate(data[start:stop], self.T)
            ds[start:stop] = self._integrate(vels[start:stop], self.T)

        return vels, ds

    # Used only for saving figures about a given recording. Completes _transform for the entire provided data
    # instead of based on the transitions and self.offset
    def full_transform(self, data):
        # Get all transitions from the data

        T = data.time.diff().mean() / 1000.
        acc = self._find_dir(data)
        rest_states = self._find_rest_states(data)

        # Apply double integration to each
        vels = np.zeros(acc.shape)
        ds = np.zeros(acc.shape)
        # Apply the integration to each compression separately
        for start, stop in zip(rest_states[:-1], rest_states[1:]):
            vels[start:stop] = self._integrate(acc[start:stop], T)
            ds[start:stop] = self._integrate(vels[start:stop], T)

        # Output the resulting velocities and displacements
        return acc, vels, ds, rest_states

    # Integrator using https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4773040/
    def _integrate(self, data, T):

        if isinstance(data, pd.Series):
            data = data.to_numpy()


        # HP filter
        b, a = butter(4, 0.6, btype='highpass', analog=False, fs=1./T, output='ba')
        data = filtfilt(b, a, data, method='gust', padtype=None)
        # LP via trapezoid rule
        data = cumtrapz(data.flatten(), dx=T, initial=0).reshape(-1,1)

        return data

    # Acceleration estimates
    # Transform the data using PCA so we obtain exclusively the press direction
    # PCA also neatly deals with the gravity constant within the problem
    def _find_dir(self, data):
        accs = data[['acc1', 'acc2', 'acc3']]
        pca = PCA(n_components=1)
        pca.fit(accs)
        return pca.transform(accs)

    def _find_rest_states(self, data):
        press1 = data.loc[:, "press1"] #- data.loc[:, "press1"].min()
        press1 = press1 / np.max(press1)
        #data = data.assign(press1=press1)
        peaks, _ = find_peaks(press1, prominence=0.05)
        return peaks


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ArduinoReader import ArduinoReader
    import time

    reader_args = {
        "port": "/dev/ttyACM0",
        "baudrate": 9600,
        "timeout": 0
    }
    reader = ArduinoReader(
        cols=["time", "press1", "press2", "press3", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"],
        save=False,
        **reader_args)

    reader.start()
    time.sleep(10)
    data = reader.get_data()
    reader.stop()

    model = Model(eps=2.0)
    accs = model._find_dir(data)
    transitions = model._find_rest_states(data)

    plt.figure()
    plt.plot(data.time, accs)
    plt.vlines(data.time[transitions], ymin=np.min(accs), ymax=np.max(accs), linestyles='dashed', colors='g')
    plt.title('Acceleration')

    plt.figure()
    plt.plot(data.time, data.press1)
    plt.vlines(data.time[transitions], ymin=np.min(data.press1), ymax=np.max(data.press1), linestyles='dashed', colors='g')
    plt.title('Pressure 1')

    plt.show()

    print(model.estimate(data))
