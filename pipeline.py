import numpy as np
import json
from typing import Dict, List
from suaBibSignal import signalMeu
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy import signal as sg

fs = 48000  # pontos por segundo (frequência de amostragem)
A = 5  # Amplitude
T = 5  # Tempo em que o seno será gerado
t = np.linspace(-T / 2, T / 2, T * fs)


def pipeline():

    yAudio, samplerate = signalMeu().read("camFis.wav")
    samplesAudio = len(yAudio)

    generate_graphic(
        t := np.arange(0, samplesAudio / samplerate, 1 / samplerate)[:-1],
        amplitude := yAudio[:, 1],
        "Tempo em segundos",
        "Amplitude",
        "Amplitude por tempo do áudio original",
    )
    sound = [[x, y] for x, y in zip(t, amplitude)]
    signalMeu().play_sound(sound, fs)

    normalized_amplitude = normalize(yAudio[:, 1])
    generate_graphic(
        t,
        normalized_amplitude,
        "Tempo em segundos",
        "Amplitude normalizada",
        "Amplitude por tempo do áudio normalizado",
    )
    sound_normalized = [[x, y] for x, y in zip(t, normalized_amplitude)]
    signalMeu().play_sound(sound_normalized, fs)

    filtered_amplitude = LPF(normalized_amplitude, 4000, fs)

    sound_filtered = [[x, y] for x, y in zip(t, filtered_amplitude)]
    generate_graphic(
        t,
        filtered_amplitude,
        "Tempo em segundos",
        "Amplitude normalizada",
        "Amplitude por tempo do áudio filtado até 4000 Hz",
    )
    signalMeu().play_sound(sound_filtered, fs)

    sin_x, sin_amp = generateSin(14000, 5, fs)
    am_amplitude = sin_amp[: len(filtered_amplitude)] * filtered_amplitude
    fequencie_four, amplitude_four = signalMeu().calcFFT(am_amplitude, fs)
    generate_graphic(
        fequencie_four,
        amplitude_four,
        "frequência em Hz",
        "Amplitude",
        "Amplitude por freqência da onda modulada em 14000 Hz",
    )
    sound_am = [[x, y] for x, y in zip(t, am_amplitude)]
    signalMeu().play_sound(sound_am, fs)

    demodulated_amplitude = [
        x / y if y != 0 else x / 0.001
        for x, y in zip(am_amplitude, sin_amp[: len(am_amplitude)])
    ]
    generate_graphic(
        t,
        demodulated_amplitude,
        "Tempo em segundos",
        "Amplitude",
        "Amplitude por tempo desmodulada",
    )
    fequencie_four_dem, amplitude_four_dem = signalMeu().calcFFT(
        demodulated_amplitude, fs
    )
    generate_graphic(
        fequencie_four_dem,
        amplitude_four_dem,
        "frequência em Hz",
        "Amplitude",
        "Amplitude por freqência desmodulada",
    )
    sound_demodulated = [[x, y] for x, y in zip(t, demodulated_amplitude)]
    signalMeu().play_sound(sound_demodulated, fs)


def generate_graphic(x, y, xlabel, ylabel, title):

    plt.plot(x, y)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=10)
    plt.grid()
    plt.show()


def normalize(signal):

    min_value = min(signal)
    max_value = max(signal)
    divided_value = max_value if max_value >= abs(min_value) else abs(min_value)

    return [value / abs(divided_value) for value in signal]


def LPF(signal, cutoff_hz, fs):

    nyq_rate = fs / 2
    width = 5.0 / nyq_rate
    ripple_db = 120.0
    N, beta = sg.kaiserord(ripple_db, width)
    taps = sg.firwin(N, cutoff_hz / nyq_rate, window=("kaiser", beta))
    return sg.lfilter(taps, 1.0, signal)


def generateSin(freq, time, fs):

    n = time * fs
    x = np.linspace(0.0, time, n)
    s = np.sin(freq * x * 2 * np.pi)
    return (x, s)


