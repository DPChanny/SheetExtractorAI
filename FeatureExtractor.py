import sys

import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from Statics import save_plot, y_sum
import math


class Sample:
    def __init__(self, sample_name):
        self.bpm = 120
        self.sample_name = sample_name
        self.amplitudes, self.sampling_rate = librosa.load("src/" + sample_name + ".wav")
        self.sample_time = librosa.get_duration(y=self.amplitudes, sr=self.sampling_rate)


class FeatureExtractor:
    def __init__(self, sample):
        self.sample = sample
        self.win_length = int(self.sample.sampling_rate * 0.1)
        self.hop_length = self.win_length // 5
        self.n_fft = 4096

    def save_spectrum(self, spectrum, y_axis, directory_name, spectrum_name):
        librosa.display.specshow(spectrum,
                                 sr=self.sample.sampling_rate,
                                 win_length=self.win_length,
                                 hop_length=self.hop_length,
                                 n_fft=self.n_fft,
                                 y_axis=y_axis,
                                 x_axis='time')
        plt.colorbar(format='%2.0f dB')
        save_plot(directory_name, spectrum_name)

    def extract_wave_feature(self, start, end, beat_division=16):
        amplitudes = self.sample.amplitudes[start:end]

        amplitudes_peaks, _ = find_peaks(np.clip(amplitudes, 0, np.inf))

        plt.plot(amplitudes, linewidth=0.05)
        save_plot(
            self.sample.sample_name + "/" + self.sample.sample_name + "_wave",
            self.sample.sample_name + "_wave " + str((start, end)))

        plt.plot(amplitudes_peaks,
                 amplitudes[amplitudes_peaks], linewidth=0.05)
        plt.scatter(amplitudes_peaks,
                    amplitudes[amplitudes_peaks], s=0.05)

        maxes = []
        max_range = math.trunc(self.sample.sampling_rate / (self.sample.bpm / 60) / beat_division)

        for j in range(0, math.trunc(len(amplitudes) / max_range)):
            maxes.append(max(amplitudes[
                             j * max_range:
                             min(j * max_range + max_range, len(amplitudes))]))

        plt.plot(
            np.linspace(start=0, stop=len(amplitudes), num=len(maxes)),
            maxes, linewidth=0.5, label="max")

        averages = []
        average_range = math.trunc(self.sample.sampling_rate / (self.sample.bpm / 60) / beat_division)

        for j in range(0, math.trunc(len(amplitudes) / average_range)):
            range_amplitudes = amplitudes[
                               j * max_range:
                               min(j * max_range + max_range, len(amplitudes))]
            range_amplitudes_peaks, _ = find_peaks(range_amplitudes)
            if len(range_amplitudes_peaks) != 0:
                averages.append(sum(range_amplitudes[range_amplitudes_peaks]) / len(range_amplitudes_peaks))
            else:
                averages.append(0)

        plt.plot(
            np.linspace(start=0, stop=len(amplitudes), num=len(averages)),
            averages, linewidth=0.5, label="average")
        plt.legend()
        save_plot(self.sample.sample_name + "/" + self.sample.sample_name + "_wave_clipped_peaks",
                  self.sample.sample_name + "_wave_clipped_peaks " + str((start, end)))

        return maxes, averages

    def extract_wave_features(self, amplitude_range=sys.maxsize):
        maxes = []
        averages = []
        amplitude_range = min(amplitude_range, len(self.sample.amplitudes))
        for i in range(0, math.trunc(len(self.sample.amplitudes) / amplitude_range)):
            wave_feature = self.extract_wave_feature(
                                i * amplitude_range,
                                min(i * amplitude_range + amplitude_range, len(self.sample.amplitudes)))
            maxes.append(wave_feature[0])
            averages.append(wave_feature[1])
        return maxes, averages

    def extract_stft_feature(self, start=0, end=sys.maxsize):
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

        magnitudes_db_y_sum = y_sum(magnitudes_db)
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(start=0,
                             stop=self.sample.sample_time * len(amplitudes) / len(self.sample.amplitudes),
                             num=len(magnitudes_db_y_sum)),
                 magnitudes_db_y_sum, linewidth=0.5)
        plt.subplot(2, 1, 2)
        self.save_spectrum(
            magnitudes_db,
            'log',
            self.sample.sample_name + "/" + self.sample.sample_name + "_stft_log",
            self.sample.sample_name + "_stft_log " + str((start, end)))

        magnitudes_mel_db_y_sum = y_sum(magnitudes_mel_db)
        plt.subplot(2, 1, 1)
        plt.plot(np.linspace(start=0,
                             stop=self.sample.sample_time,
                             num=len(magnitudes_mel_db_y_sum)),
                 magnitudes_mel_db_y_sum, linewidth=0.5)
        plt.subplot(2, 1, 2)
        self.save_spectrum(
            magnitudes_mel_db,
            'mel',
            self.sample.sample_name + "/" + self.sample.sample_name + "_stft_mel",
            self.sample.sample_name + "_stft_mel " + str((start, end)))

        return magnitudes_mel_db, magnitudes_mel_db_y_sum

    def extract_stft_features(self, amplitude_range=sys.maxsize):
        magnitudes_mel_dbs = []
        magnitudes_mel_db_y_sums = []
        amplitude_range = min(amplitude_range, len(self.sample.amplitudes))
        for i in range(0, math.trunc(len(self.sample.amplitudes) / amplitude_range)):
            stft_feature_range = self.extract_stft_feature(
                            i * amplitude_range,
                            min(i * amplitude_range + amplitude_range, len(self.sample.amplitudes)))
            magnitudes_mel_dbs.append(stft_feature_range[0])
            magnitudes_mel_db_y_sums.append(stft_feature_range[1])
        return magnitudes_mel_dbs, magnitudes_mel_db_y_sums
    