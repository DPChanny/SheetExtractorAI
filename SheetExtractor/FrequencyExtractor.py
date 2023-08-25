import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import math

WIN_LENGTH = 2048
HOP_LENGTH = WIN_LENGTH // 4


def find_max_dense_frequency_position(frequencies, size):
    max_dense = min(frequencies) * size
    max_dense_frequency_position = 0
    for i in range(0, len(frequencies) - size):
        current_dense = sum(frequencies[i: i + size])
        if max_dense < current_dense:
            max_dense = current_dense
            max_dense_frequency_position = i
    return max_dense_frequency_position + size // 2


def unify_beat(beat):
    min_error = math.inf
    min_error_frequency = 0
    for i in range(0, len(beat)):
        current_error = 0
        for j in range(0, len(beat)):
            current_error += abs(beat[j] - beat[i])
        if current_error < min_error:
            min_error = current_error
            min_error_frequency = beat[i]
    return np.repeat(min_error_frequency, len(beat)), min_error_frequency


def unify_frequencies(frequencies, bar_count, beat_count_per_bar):
    unified_frequencies = np.zeros(len(frequencies))
    result_frequencies = []
    beats = np.linspace(start=0, stop=len(frequencies), num=bar_count * beat_count_per_bar + 1)
    for i in range(0, len(beats) - 1):
        unified_frequency, result_frequency = unify_beat(frequencies[round(beats[i]): round(beats[i + 1])])
        unified_frequencies[round(beats[i]): round(beats[i + 1])] = unified_frequency
        result_frequencies.append(result_frequency)
    return unified_frequencies, result_frequencies


def extract_frequencies(file_name, bar_count, beat_count_per_bar):
    amplitudes, sampling_rate = librosa.load(file_name + ".wav")
    stft = librosa.amplitude_to_db(abs(librosa.stft(amplitudes,
                                                    n_fft=WIN_LENGTH,
                                                    win_length=WIN_LENGTH,
                                                    hop_length=HOP_LENGTH)),
                                   ref=np.max)
    librosa.display.specshow(stft,
                             sr=sampling_rate,
                             hop_length=HOP_LENGTH,
                             y_axis='linear',
                             x_axis='time')
    plt.colorbar(format='%2.0f dB')
    frequencies = []

    for x in range(0, stft.shape[1]):
        frequencies.append((sampling_rate / 2) * (find_max_dense_frequency_position(stft[:, x], 5) / stft.shape[0]))

    unified_frequencies, result_frequencies = unify_frequencies(frequencies, bar_count, beat_count_per_bar)

    plt.plot(np.linspace(start=0,
                         stop=librosa.get_duration(amplitudes, sampling_rate),
                         num=len(frequencies)),
             frequencies, 'g')
    plt.plot(np.linspace(start=0,
                         stop=librosa.get_duration(amplitudes, sampling_rate),
                         num=len(unified_frequencies)),
             unified_frequencies, 'b')

    plt.ylim(0, 2000)
    plt.savefig(file_name + ".png", dpi=500)
    plt.clf()
    print(result_frequencies)
    return result_frequencies
