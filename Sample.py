from os.path import join
from sys import maxsize as int_max
from Public import SOURCE
from librosa import load, get_duration


class Sample:
    def __init__(self, directory: list[str], sample_name: str, beat_per_minute: int, start: int = 0, end: int = int_max, log: bool = False):
        self.beat_per_minute = beat_per_minute
        self.beat_per_second = self.beat_per_minute / 60

        self.sample_name = sample_name

        directory = join(*directory)
        if log:
            print("Loading " + join(*[SOURCE, directory, sample_name + ".wav"]))
        amplitudes, self.sampling_rate = load(join(*[SOURCE, directory, sample_name + ".wav"]))
        self.amplitudes = amplitudes[start:min(end, len(amplitudes))]
        self.duration = get_duration(y=self.amplitudes, sr=self.sampling_rate)

        self.win_length = int(self.sampling_rate / self.beat_per_second / 16)
        self.hop_length = int(self.win_length / 4)
        self.n_fft = 4096
