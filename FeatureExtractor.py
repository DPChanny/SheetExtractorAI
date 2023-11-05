import sys
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.signal import find_peaks
import Sample
from Statics import save_plot


class STFTFeature:
    def __init__(self,
                 magnitudes_db: ndarray,
                 magnitudes_mel_db: ndarray,
                 magnitudes_sum: ndarray,
                 duration: float):
        self.magnitudes_db = magnitudes_db
        self.magnitudes_mel_db = magnitudes_mel_db
        self.magnitudes_sum = magnitudes_sum
        self.duration = duration


class FeatureExtractor:
    def __init__(self, sample: Sample):
        self.sample = sample
        self.win_length = int(self.sample.sampling_rate / self.sample.beat_per_second / 16)
        self.hop_length = int(self.win_length / 4)
        self.n_fft = 2048

    def save_spectrum(self, spectrum, y_axis: str, directory_name: str, spectrum_name: str):
        librosa.display.specshow(spectrum,
                                 sr=self.sample.sampling_rate,
                                 win_length=self.win_length,
                                 hop_length=self.hop_length,
                                 n_fft=self.n_fft,
                                 y_axis=y_axis,
                                 x_axis='time')
        plt.colorbar(format='%2.0f dB')
        save_plot(directory_name, spectrum_name, spectrum.shape)

    # 샘플 파형을 start 에서 end 까지 분석
    def extract_wave_feature(self, start: int = 0, end: int = sys.maxsize, plot: bool = False):
        end = min(end, len(self.sample.amplitudes))
        amplitudes = self.sample.amplitudes[start:end]

        amplitudes_peaks, _ = find_peaks(np.clip(amplitudes, 0, np.inf))

        if plot:
            plt.plot(amplitudes, linewidth=0.05)
            save_plot(self.sample.sample_name + "/" + self.sample.sample_name + "_wave",
                      self.sample.sample_name + "_wave " + str((start, end)), str(len(amplitudes)))

            plt.plot(amplitudes_peaks,
                     amplitudes[amplitudes_peaks], linewidth=0.05)
            plt.scatter(amplitudes_peaks,
                        amplitudes[amplitudes_peaks], s=0.05)
            save_plot(self.sample.sample_name + "/" + self.sample.sample_name + "_wave_clipped_peaks",
                      self.sample.sample_name + "_wave_clipped_peaks " + str((start, end)), str(len(amplitudes_peaks)))

    # 샘플 파형 전체를 division_range 만큼 묶어서 분석
    def extract_wave_features(self, division_range: int = sys.maxsize, plot: bool = False):
        division_range = min(division_range, len(self.sample.amplitudes))
        for i in range(int(len(self.sample.amplitudes) / division_range)):
            self.extract_wave_feature(i * division_range,
                                      min(i * division_range + division_range, len(self.sample.amplitudes)),
                                      plot)

    # 샘플 주파수를 start 부터 end 까지 분석
    # 샘플 주파수의 STFT Db(log), STFT MEL Db(log), STFT Sum 을 반환
    def extract_stft_feature(self, start: int = 0, end: int = sys.maxsize, plot: bool = False) -> STFTFeature:
        end = min(end, len(self.sample.amplitudes))
        amplitudes = self.sample.amplitudes[start:end]

        print("STFT:", self.sample.sample_name, str((start, end)))
        amplitudes_stft = librosa.stft(amplitudes,
                                       win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       n_fft=self.n_fft)

        magnitudes = abs(amplitudes_stft)
        magnitudes_db = librosa.amplitude_to_db(magnitudes, ref=np.max)
        print("RESULT:", str(magnitudes.shape))

        print("MEL:", self.sample.sample_name, str((start, end)))
        magnitudes_mel = librosa.feature.melspectrogram(S=magnitudes,
                                                        sr=self.sample.sampling_rate,
                                                        win_length=self.win_length,
                                                        hop_length=self.hop_length,
                                                        n_fft=self.n_fft)
        magnitudes_mel_db = librosa.amplitude_to_db(magnitudes_mel, ref=np.max)
        print("RESULT: " + str(magnitudes_mel_db.shape))

        if plot:
            self.save_spectrum(magnitudes_db,
                               'log',
                               self.sample.sample_name + "/" + self.sample.sample_name + "_stft_log",
                               self.sample.sample_name + "_stft_log " + str((start, end)))

            self.save_spectrum(magnitudes_mel_db,
                               'mel',
                               self.sample.sample_name + "/" + self.sample.sample_name + "_stft_mel",
                               self.sample.sample_name + "_stft_mel " + str((start, end)))

        magnitudes_sum = []
        for i in range(magnitudes.shape[1]):
            magnitudes_sum.append(sum(magnitudes[:, i]))

        if plot:
            plt.plot(np.linspace(start=0,
                                 stop=self.sample.sample_time * len(amplitudes) / len(self.sample.amplitudes),
                                 num=len(magnitudes_sum)),
                     magnitudes_sum, linewidth=0.5)
            save_plot(self.sample.sample_name + "/" + self.sample.sample_name + "_stft_sum",
                      self.sample.sample_name + "_stft_sum " + str((start, end)), str(len(magnitudes_sum)))

        return STFTFeature(magnitudes_db,
                           magnitudes_mel_db,
                           np.array(magnitudes_sum),
                           self.sample.sample_time * len(amplitudes) / len(self.sample.amplitudes))

    # 샘플 주파수 전체를 division_range 만큼 묶어서 분석
    # 각 샘플 주파수 묶음의 STFT Db(log), STFT MEL Db(log), STFT Sum 을 반환
    def extract_stft_features(self, division_range: int = sys.maxsize, plot: bool = False) -> list[STFTFeature]:
        stft_features = []
        division_range = min(division_range, len(self.sample.amplitudes))
        for i in range(int(len(self.sample.amplitudes) / division_range)):
            stft_features.append(
                self.extract_stft_feature(i * division_range,
                                          min(i * division_range + division_range, len(self.sample.amplitudes)),
                                          plot))
        return stft_features
    