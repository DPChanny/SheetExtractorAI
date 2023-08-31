import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from Statics import save_plot


class Sample:
    def __init__(self, sample_name):
        self.sample_name = sample_name
        self.amplitudes, self.sampling_rate = librosa.load("src/" + sample_name + ".wav")
        self.sample_time = librosa.get_duration(y=self.amplitudes, sr=self.sampling_rate)


class FeatureExtractor:
    def __init__(self, sample):
        self.sample = sample

    def extract_wave_feature(self):
        amplitudes_clipped = np.clip(self.sample.amplitudes, 0, np.inf)
        amplitudes_clipped_peaks, _ = find_peaks(amplitudes_clipped)

        plt.plot(self.sample.amplitudes, linewidth=0.1)
        save_plot(self.sample.sample_name, self.sample.sample_name + "_wave")

        plt.plot(amplitudes_clipped, linewidth=0.1)
        save_plot(self.sample.sample_name, self.sample.sample_name + "_wave_clipped")

        plt.plot(amplitudes_clipped_peaks,
                 amplitudes_clipped[amplitudes_clipped_peaks], linewidth=0.1)
        save_plot(self.sample.sample_name, self.sample.sample_name + "_wave_clipped_peaks")

    def extract_stft_feature(self):
        win_length = int(self.sample.sampling_rate * 0.05)
        hop_length = win_length // 5
        n_fft = 4096

        amplitudes_stft = librosa.stft(self.sample.amplitudes,
                                       win_length=win_length,
                                       hop_length=hop_length,
                                       n_fft=n_fft)

        magnitudes = abs(amplitudes_stft)
        magnitudes_db = librosa.amplitude_to_db(magnitudes, ref=np.max)

        magnitudes_mel = librosa.feature.melspectrogram(S=magnitudes,
                                                        sr=self.sample.sampling_rate,
                                                        win_length=win_length,
                                                        hop_length=hop_length,
                                                        n_fft=n_fft)
        magnitudes_mel_db = librosa.amplitude_to_db(magnitudes_mel, ref=np.max)

        def y_sum(array):
            result = []
            for i in range(0, array.shape[1]):
                result.append(sum(abs(array[:, i])))
            return result

        def save_spectrum(spectrum, y_axis, spectrum_name):
            librosa.display.specshow(spectrum,
                                     sr=self.sample.sampling_rate,
                                     win_length=win_length,
                                     hop_length=hop_length,
                                     n_fft=n_fft,
                                     y_axis=y_axis,
                                     x_axis='time')
            plt.colorbar(format='%2.0f dB')
            plt.title(spectrum.shape)
            spectrum_y_sum = y_sum(spectrum)
            plt.plot(np.linspace(start=0,
                                 stop=self.sample.sample_time,
                                 num=len(spectrum_y_sum)),
                     spectrum_y_sum, linewidth=0.1)
            save_plot(self.sample.sample_name, spectrum_name)

        # save_spectrum(magnitudes_db, 'linear', self.sample.sample_name + "_stft_log_linear")
        save_spectrum(magnitudes_db, 'log', self.sample.sample_name + "_stft_log_log")

        # save_spectrum(magnitudes_mel_db, 'linear', self.sample.sample_name + "_stft_mel_linear")
        save_spectrum(magnitudes_mel_db, 'mel', self.sample.sample_name + "_stft_mel_mel")
