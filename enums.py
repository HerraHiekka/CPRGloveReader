from enum import Enum

class PressState(Enum):
    NO_PRESS = 0
    PRESS = 1

class State(Enum):
    INTERRUPTION = 'interrupted'
    READY = 'ready'
    OK = 'ok'
    LOW_RATE = 'rateLow'
    HIGH_RATE = 'rateHigh'
    LOW_DEPTH = 'depthLow'
    HIGH_DEPTH = 'depthHigh'
    LEAN = 'noRecoil'