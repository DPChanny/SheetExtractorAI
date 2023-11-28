import librosa.feature
import matplotlib.pyplot as plt
from numpy import ndarray, array, linspace, clip
from scipy.signal import find_peaks
from Sample import Sample
from LoadSave import save_plot


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


def save_spectrum_plot(sample: Sample,
                       spectrum,
                       y_axis: str,
                       directory_name: str,
                       plot_name: str,
                       plot_title: str,
                       log: bool = False):
    librosa.display.specshow(spectrum,
                             sr=sample.sampling_rate,
                             win_length=sample.win_length,
                             hop_length=sample.hop_length,
                             n_fft=sample.n_fft,
                             y_axis=y_axis,
                             x_axis='time')
    plt.colorbar(format='%2.0f dB')
    save_plot(directory_name, plot_name, plot_title, log)


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
                           plot_name: str):
    plt.plot(wave_feature.amplitudes, linewidth=0.05)
    save_plot(directory_name,
              plot_name + ".wfa",
              sample.name + " Wave Feature: Amplitudes")

    plt.plot(wave_feature.amplitudes_peaks,
             sample.amplitudes[wave_feature.amplitudes_peaks], linewidth=0.05)
    plt.scatter(wave_feature.amplitudes_peaks,
                sample.amplitudes[wave_feature.amplitudes_peaks], s=0.05)
    save_plot(directory_name,
              plot_name + ".wfap",
              sample.name + "Wave Feature: Amplitudes Peaks")


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
    save_spectrum_plot(sample,
                       stft_feature.magnitudes_db,
                       'log',
                       directory_name,
                       plot_name + ".sfl",
                       sample.name + " STFT Feature: Magnitudes dB",
                       log)

    save_spectrum_plot(sample,
                       stft_feature.magnitudes_mel_db,
                       'mel',
                       directory_name,
                       sample.name + ".sfm",
                       sample.name + " STFT Feature: Magnitudes Mel dB",
                       log)

    plt.plot(linspace(start=0,
                      stop=sample.duration * len(sample.amplitudes) / len(sample.amplitudes),
                      num=len(stft_feature.magnitudes_sum)),
             stft_feature.magnitudes_sum, linewidth=0.5)

    save_plot(directory_name,
              plot_name + ".sfs",
              sample.name + " STFT Feature: Magnitudes Sum",
              log)
