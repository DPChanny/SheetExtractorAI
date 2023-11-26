from Statics import SOURCE
from librosa import load, get_duration


class Sample:
    def __init__(self, sample_name: str, sample_beat_per_minute: int):
        self.beat_per_minute = sample_beat_per_minute
        self.beat_per_second = self.beat_per_minute / 60
        self.sample_name = sample_name
        self.amplitudes, self.sampling_rate = load("./" + SOURCE + "/" + sample_name + ".wav")
        self.sample_time = get_duration(y=self.amplitudes, sr=self.sampling_rate)
