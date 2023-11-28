import sys

from Public import SOURCE
from librosa import load, get_duration


class Sample:
    def __init__(self, name: str, beat_per_minute: int, start: int = 0, end: int = sys.maxsize, log: bool = False):
        self.beat_per_minute = beat_per_minute
        self.beat_per_second = self.beat_per_minute / 60

        self.name = name

        if log:
            print("Loading " + name)
        amplitudes, self.sampling_rate = load("./" + SOURCE + "/" + name + ".wav")
        self.amplitudes = amplitudes[start:min(end, len(amplitudes))]
        self.duration = get_duration(y=self.amplitudes, sr=self.sampling_rate)

        self.win_length = int(self.sampling_rate / self.beat_per_second / 16)
        self.hop_length = int(self.win_length / 4)
        self.n_fft = 4096
