import os
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np


sample_name = "sample"


def save_plot(plot_name):
    os.makedirs("src/" + sample_name + "/", exist_ok=True)
    plt.savefig("src/" + sample_name + "/" + plot_name + ".png", dpi=1000)
    plt.clf()


amplitudes, sampling_rate = librosa.load("src/" + sample_name + ".wav")

plt.plot(amplitudes)
save_plot(sample_name + "_wave")

win_length = int(sampling_rate * 0.05)
hop_length = win_length // 5
n_fft = 4096

print(len(amplitudes), sampling_rate, win_length, hop_length)

amplitudes_stft = librosa.stft(amplitudes,
                               win_length=win_length,
                               hop_length=hop_length,
                               n_fft=n_fft)

magnitudes = abs(amplitudes_stft)
magnitudes_db = librosa.amplitude_to_db(magnitudes, ref=np.max)

magnitudes_mel = librosa.feature.melspectrogram(S=magnitudes,
                                                win_length=win_length,
                                                hop_length=hop_length,
                                                sr=sampling_rate,
                                                n_fft=n_fft)
magnitudes_mel_db = librosa.amplitude_to_db(magnitudes_mel, ref=np.max)


def save_spectrum(spectrum, y_axis, spectrum_name):
    librosa.display.specshow(spectrum,
                             sr=sampling_rate,
                             y_axis=y_axis,
                             x_axis='time')
    plt.colorbar(format='%2.0f dB')
    plt.title(spectrum.shape)
    save_plot(spectrum_name)


save_spectrum(magnitudes_db, 'log', sample_name + "_stft_log_log")
save_spectrum(magnitudes_db, 'linear', sample_name + "_stft_log_linear")

save_spectrum(magnitudes_mel_db, 'mel', sample_name + "_stft_mel_mel")
save_spectrum(magnitudes_mel_db, 'linear', sample_name + "_stft_mel_linear")
