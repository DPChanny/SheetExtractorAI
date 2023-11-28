import librosa.feature
import matplotlib.pyplot as plt
from numpy import ndarray, array, clip
from scipy.signal import find_peaks
from Sample import Sample
from Public import save_plot, set_tick


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


# 샘플 파형을 start 에서 end 까지 분석
def extract_wave_feature(sample: Sample, log: bool = False) -> WaveFeature:
    if log:
        print("Extracting " + sample.name + " wave feature")

    amplitudes = sample.amplitudes
    amplitudes_peaks, _ = find_peaks(clip(amplitudes, 0, max(abs(amplitudes))))

    return WaveFeature(amplitudes, amplitudes_peaks)


def save_wave_feature_plot(sample: Sample,
                           wave_feature: WaveFeature,
                           directory_name: str,
                           plot_name: str,
                           log: bool = False):
    fig = plt.figure(figsize=(sample.duration * sample.beat_per_second * 2, 10))
    fig.suptitle(sample.name + " Wave Feature")

    amplitudes_ax = plt.subplot(211)
    amplitudes_ax.set_title("Amplitudes")
    amplitudes_ax.plot(wave_feature.amplitudes, linewidth=0.05)
    set_tick(amplitudes_ax,
             (0, len(wave_feature.amplitudes), sample.sampling_rate / sample.beat_per_second / 4),
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
             (0, len(wave_feature.amplitudes), sample.sampling_rate / sample.beat_per_second / 4),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    fig.tight_layout()
    save_plot(directory_name, plot_name + ".wf", fig, log=log)


# 샘플 주파수를 start 부터 end 까지 분석
# 샘플 주파수의 STFT Db(log), STFT MEL Db(log), STFT Sum 을 반환
def extract_stft_feature(sample: Sample, log: bool = False) -> STFTFeature:
    if log:
        print("Extracting " + sample.name + " stft feature")

    amplitudes_stft = librosa.stft(sample.amplitudes,
                                   win_length=sample.win_length,
                                   hop_length=sample.hop_length,
                                   n_fft=sample.n_fft)

    magnitudes = abs(amplitudes_stft)
    magnitudes_db = librosa.amplitude_to_db(magnitudes)

    magnitudes_mel = librosa.feature.melspectrogram(S=magnitudes,
                                                    sr=sample.sampling_rate,
                                                    win_length=sample.win_length,
                                                    hop_length=sample.hop_length,
                                                    n_fft=sample.n_fft)
    magnitudes_mel_db = librosa.amplitude_to_db(magnitudes_mel)

    magnitudes_sum = []
    for i in range(magnitudes.shape[1]):
        magnitudes_sum.append(sum(magnitudes[:, i]))

    return STFTFeature(magnitudes_db,
                       magnitudes_mel_db,
                       array(magnitudes_sum))


def save_stft_feature_plot(sample: Sample,
                           stft_feature: STFTFeature,
                           directory_name: str,
                           plot_name: str,
                           log: bool = False):
    fig = plt.figure(figsize=(sample.duration * sample.beat_per_second * 2, 15))
    fig.suptitle(sample.name + " STFT Feature")

    magnitudes_db_ax = fig.add_subplot(311)
    magnitudes_db_ax.set_title("Magnitudes dB")
    librosa.display.specshow(stft_feature.magnitudes_db,
                             sr=sample.sampling_rate,
                             win_length=sample.win_length,
                             hop_length=sample.hop_length,
                             n_fft=sample.n_fft,
                             y_axis="log",
                             ax=magnitudes_db_ax)
    set_tick(magnitudes_db_ax,
             (0, stft_feature.magnitudes_db.shape[1], 5),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    magnitudes_mel_db_ax = fig.add_subplot(312)
    magnitudes_mel_db_ax.set_title("Magnitudes Mel dB")
    librosa.display.specshow(stft_feature.magnitudes_mel_db,
                             sr=sample.sampling_rate,
                             win_length=sample.win_length,
                             hop_length=sample.hop_length,
                             n_fft=sample.n_fft,
                             y_axis="mel",
                             ax=magnitudes_mel_db_ax)
    set_tick(magnitudes_mel_db_ax,
             (0, stft_feature.magnitudes_mel_db.shape[1], 5),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    magnitudes_sum_ax = fig.add_subplot(313)
    magnitudes_sum_ax.set_title("Magnitudes Sum")
    plt.plot(stft_feature.magnitudes_sum, linewidth=0.5)
    set_tick(magnitudes_sum_ax,
             (0, len(stft_feature.magnitudes_sum), 5),
             (0, sample.duration, 1 / sample.beat_per_second / 4))

    fig.tight_layout()
    save_plot(directory_name, plot_name + ".sf", fig, log=log)
