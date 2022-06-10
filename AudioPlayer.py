from playsound import playsound
from gtts import gTTS

import yaml
from glob import glob

import os


class AudioPlayer:

    def __init__(self, phrases=f'phrases{os.path.sep}en-US_phrases.yaml', lang='en'):
        AudioPlayer._generate_sounds(phrases, lang=lang)

    # Playing the sounds should probably be blocking to avoid starting a new feedback while previous is still playing.
    @staticmethod
    def play(audio):
        if os.path.exists(f'audio{os.path.sep}{audio}.mp3'):
            playsound(f'audio{os.path.sep}{audio}.mp3')
        else:
            print(f"Could not fine audio{os.path.sep}{audio}.mp3")
            playsound(f'audio/missingSound.mp3')
        if "rate" in audio:
            playsound(f'audio/metronome.mp3')

    @staticmethod
    def _generate_sounds(phrases, lang):
        # Clear the old sounds
        files = glob(f'audio{os.path.sep}*.mp3')
        for file in files:
            if "metronome" not in file:
                os.remove(file)

        # Generate the audio library from phrases
        with open(phrases, 'r') as file:
            phraseList = yaml.load(file, Loader=yaml.FullLoader)

        keys = list(phraseList.keys())
        for key in keys:
            sound = gTTS(text=phraseList[key], lang=lang, slow=False)
            sound.save(f'audio{os.path.sep}{key}.mp3')


if __name__ == '__main__':
    AudioPlayer(phrases='phrases/fi-FI_phrases.yaml', lang='fi')
    #AudioPlayer.play('noRecoil')