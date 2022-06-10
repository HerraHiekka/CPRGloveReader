from ArduinoReader import ArduinoReader
from AudioPlayer import AudioPlayer
from Model import Model
from enums import State

import yaml
import keyboard
import matplotlib.pyplot as plt
import numpy as np

import os
import time
from functools import reduce

# Maintains the state of the CPR (freq, depth, etc.)
# Provides feedback based on the state
# Main loop of the program
class CPRState:

    def __init__(self, save=None, config='./config.yaml'):

        print("Reading config")
        with open(config, 'r') as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)

        # Prepare for saving, if necessary
        if save is not None:
            # Append the time to the desired session name
            self.save = f"{save} - {int(time.time())}"
            os.mkdir(f"./recordings/{self.save}")

        # Prepare audio
        print("Preparing audio")
        phrases = f"phrases{os.path.sep}{self.config['locale']}_phrases.yaml"
        AudioPlayer(phrases=phrases, lang=self.config['locale'][:2]) # The first 2 chars of the country code specify the language for gTTS

        # Establish the connection to Arduino for data
        print("Starting Arduino connection")
        self.reader = ArduinoReader(save=self.save, **self.config['arduino'])

        print("Initializing the session")
        # Set up the model
        self.model = Model(freq_bounds=self.config["frequency bounds"],
                           depth_bounds=self.config["depth bounds"],
                           eps=self.config["pressure th"],
                           calibration=self.config["calibration"])

        # Prepare for and start the main loop
        self.state = State.READY
        self.feedback_timer = 0

    # Main loop of the program
    def run(self):
        time.sleep(self.config["initial timer"])
        AudioPlayer.play("ready")
        self.reader.start()
        # Let the user perform for a moment before giving feedback
        self.feedback_timer = time.time() #+ self.config["initial timer"]
        if self.config["session length"] is not None:
            self.ending_time = self.config["session length"] + time.time()
        else:
            self.ending_time = None
        self.state = State.OK
        try:
            while True:
                if time.time() - self.feedback_timer > self.config["feedback timer"]:
                    data = self.reader.get_data()
                    self.state = self.model.estimate(data)
                    # We give feedback and enable cooldown only if state is not OK
                    if self.state != State.OK:
                        if not self.config["calibration"]:
                            AudioPlayer.play(self.state.value)
                        self.feedback_timer = time.time()
                time.sleep(0.5)
                if self.ending_time is not None \
                        and self.ending_time < time.time():
                    break

            self.stop()

        except KeyboardInterrupt:
            self.stop()

    # Stop for graceful exit
    def stop(self):
        # Exit cleanup only done if the program is running currently
        if self.state != State.READY:
            # Save the main data
            self.reader.stop()
            if self.save is not None:
                # Create plots and save
                acc, vel, d, rests = self.model.full_transform(self.reader.data)
                dpi = 300
                size = (len(acc)/30, 8)
                # Acceleration
                plt.figure(figsize=size)
                plt.plot(self.reader.data.time[rests[3]:], acc[rests[3]:])
                plt.vlines(self.reader.data.time[rests[3:]], ymin=np.min(acc[rests[3]:]), ymax=np.max(acc[rests[3]:]), linestyles='dashed', colors='g')
                plt.title("Acceleration")
                plt.xlabel("Time (ms)")
                plt.ylabel("Acceleration ($ms^{-2}$)")
                plt.savefig(f"./recordings/{self.save}/Acceleration.png", dpi=dpi)

                # Velocity
                plt.figure(figsize=size)
                plt.plot(self.reader.data.time[rests[3]:], vel[rests[3]:])
                plt.vlines(self.reader.data.time[rests[3:]], ymin=np.min(vel[rests[3]:]), ymax=np.max(vel[rests[3]:]), linestyles='dashed', colors='g')
                plt.title("Velocity")
                plt.xlabel("Time (ms)")
                plt.ylabel("Velocity ($ms^{-1}$)")
                plt.savefig(f"./recordings/{self.save}/Velocity.png", dpi=dpi)

                # Displacement
                plt.figure(figsize=size)
                plt.plot(self.reader.data.time[rests[3]:], d[rests[3]:])
                plt.vlines(self.reader.data.time[rests[3:]], ymin=np.min(d[rests[3]:]), ymax=np.max(d[rests[3]:]), linestyles='dashed', colors='g')
                plt.title("Displacement")
                plt.xlabel("Time (ms)")
                plt.ylabel("Displacement ($ms^{-1}$)")
                plt.savefig(f"./recordings/{self.save}/Displacement.png", dpi=dpi)

                # Pressure
                press1 = self.reader.data.press1
                press2 = self.reader.data.press1
                press3 = self.reader.data.press1
                plt.figure(figsize=size)
                plt.plot(self.reader.data.time[rests[3]:], press1[rests[3]:])
                plt.plot(self.reader.data.time[rests[3]:], press2[rests[3]:])
                plt.plot(self.reader.data.time[rests[3]:], press3[rests[3]:])
                plt.vlines(self.reader.data.time[rests[3:]], ymin=np.min(press1[rests[3]:]), ymax=np.max(press1[rests[3]:]), linestyles='dashed', colors='g')
                plt.title("Pressure Sensor Readings")
                plt.xlabel("Time (ms)")
                plt.ylabel("Force (arbitrary)")
                plt.savefig(f"./recordings/{self.save}/Pressure.png", dpi=dpi)

            self.state = State.READY
            AudioPlayer.play('done')


if __name__ == '__main__':
    pass
