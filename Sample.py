from LoadSave import SOURCE
from librosa import load, get_duration


class Sample:
    def __init__(self, name: str, beat_per_minute: int):
        self.beat_per_minute = beat_per_minute
        self.beat_per_second = self.beat_per_minute / 60

        self.name = name

        self.amplitudes, self.sampling_rate = load("./" + SOURCE + "/" + name + ".wav")
        self.duration = get_duration(y=self.amplitudes, sr=self.sampling_rate)

        self.win_length = int(self.sampling_rate / self.beat_per_second / 16)
        self.hop_length = int(self.win_length / 4)
        self.n_fft = 4096
