from typing import TYPE_CHECKING

from librosa import amplitude_to_db, stft
from librosa.display import specshow
from librosa.feature import melspectrogram
from matplotlib.pyplot import figure
from numpy import array, clip, ndarray
from scipy.signal import find_peaks

from extraction.beat_extraction import plot_beats, plot_beat_states
from public import save_plot, set_tick, FIG_WIDTH_MULTIPLIER, FIG_HEIGHT

if TYPE_CHECKING:
    from extraction.beat_extraction import BeatState, Beat
    from public import Sample


class STFTFeature:
    def __init__(self,
                 magnitudes_db: ndarray,
                 magnitudes_mel_db: ndarray,
                 magnitudes_sum: ndarray):
        self.magnitudes_db = magnitudes_db
        self.magnitudes_mel_db = magnitudes_mel_db
        self.magnitudes_sum = magnitudes_sum


class WaveFeature:
    def __init__(self, amplitudes, amplitudes_peaks):
        self.amplitudes = amplitudes
        self.amplitudes_peaks = amplitudes_peaks


def extract_wave_feature(sample: "Sample", log: bool = False) -> WaveFeature:
    if log:
        print("Extracting " + sample.sample_name + " wave feature")

    amplitudes = sample.amplitudes
    amplitudes_peaks, _ = find_peaks(clip(amplitudes, 0, max(amplitudes)))

    return WaveFeature(amplitudes, amplitudes_peaks)


def save_wave_feature_plot(sample: "Sample",
                           wave_feature: WaveFeature,
                           directory: list[str],
                           plot_name: str,
                           log: bool = False):
    fig = figure(figsize=(sample.duration * sample.beat_per_second * FIG_WIDTH_MULTIPLIER, FIG_HEIGHT * 2))
    fig.suptitle(sample.sample_name + " Wave Feature")

    amplitudes_ax = fig.add_subplot(211)
    amplitudes_ax.set_title("Amplitudes")
    amplitudes_ax.plot(wave_feature.amplitudes, linewidth=0.05)
    set_tick(amplitudes_ax,
             (0, len(wave_feature.amplitudes), int(sample.sampling_rate / sample.beat_per_second / 8)),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    amplitudes_peaks_ax = fig.add_subplot(212)
    amplitudes_peaks_ax.set_title("Amplitudes Peaks")
    amplitudes_peaks_ax.plot(wave_feature.amplitudes_peaks,
                             sample.amplitudes[wave_feature.amplitudes_peaks],
                             linewidth=0.05)
    amplitudes_peaks_ax.scatter(wave_feature.amplitudes_peaks,
                                sample.amplitudes[wave_feature.amplitudes_peaks],
                                s=0.1)
    set_tick(amplitudes_peaks_ax,
             (0, len(wave_feature.amplitudes), int(sample.sampling_rate / sample.beat_per_second / 8)),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    fig.tight_layout()
    save_plot(directory, plot_name + ".wf", fig, log=log)


def extract_stft_feature(sample: "Sample", log: bool = False) -> STFTFeature:
    if log:
        print("Extracting " + sample.sample_name + " stft feature")

    amplitudes_stft = stft(sample.amplitudes,
                           win_length=sample.win_length,
                           hop_length=sample.hop_length,
                           n_fft=sample.n_fft)

    magnitudes = abs(amplitudes_stft)
    magnitudes_db = amplitude_to_db(magnitudes)

    magnitudes_mel = melspectrogram(S=magnitudes,
                                    sr=sample.sampling_rate,
                                    win_length=sample.win_length,
                                    hop_length=sample.hop_length,
                                    n_fft=sample.n_fft)
    magnitudes_mel_db = amplitude_to_db(magnitudes_mel)

    magnitudes_sum = []
    for i in range(magnitudes.shape[1]):
        magnitudes_sum.append(sum(magnitudes[:, i]))

    return STFTFeature(magnitudes_db,
                       magnitudes_mel_db,
                       array(magnitudes_sum))


def save_stft_feature_plot(sample: "Sample",
                           stft_feature: STFTFeature,
                           directory: list[str],
                           plot_name: str,
                           beats: list["Beat"] = None,
                           beat_states: list["BeatState"] = None,
                           log: bool = False):
    fig = figure(figsize=(sample.duration * sample.beat_per_second * FIG_WIDTH_MULTIPLIER, FIG_HEIGHT * 3))
    fig.suptitle(sample.sample_name + " STFT Feature")

    magnitudes_db_ax = fig.add_subplot(311)
    magnitudes_db_ax.set_title("Magnitudes dB")
    specshow(stft_feature.magnitudes_db,
             sr=sample.sampling_rate,
             win_length=sample.win_length,
             hop_length=sample.hop_length,
             n_fft=sample.n_fft,
             y_axis="log",
             ax=magnitudes_db_ax)
    if beats is not None:
        plot_beats(magnitudes_db_ax, beats)
    set_tick(magnitudes_db_ax,
             (0, stft_feature.magnitudes_db.shape[1], 5),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    magnitudes_mel_db_ax = fig.add_subplot(312)
    magnitudes_mel_db_ax.set_title("Magnitudes Mel dB")
    specshow(stft_feature.magnitudes_mel_db,
             sr=sample.sampling_rate,
             win_length=sample.win_length,
             hop_length=sample.hop_length,
             n_fft=sample.n_fft,
             y_axis="mel",
             ax=magnitudes_mel_db_ax)
    if beats is not None:
        plot_beats(magnitudes_mel_db_ax, beats)
    set_tick(magnitudes_mel_db_ax,
             (0, stft_feature.magnitudes_mel_db.shape[1], 5),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    magnitudes_sum_ax = fig.add_subplot(313)
    magnitudes_sum_ax.set_title("Magnitudes Sum")
    magnitudes_sum_ax.plot(stft_feature.magnitudes_sum, linewidth=0.25, color="black")
    if beats is not None:
        plot_beats(magnitudes_sum_ax, beats)
    if beat_states is not None:
        plot_beat_states(magnitudes_sum_ax, beat_states, stft_feature)
    set_tick(magnitudes_sum_ax,
             (0, len(stft_feature.magnitudes_sum), 5),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    fig.tight_layout()
    save_plot(directory, plot_name + ".sf", fig, log=log)
