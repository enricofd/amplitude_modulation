import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal as window
from scipy.fftpack import fft, fftshift


class signalMeu:
    def __init__(self):
        self.init = 0

    def __init__(self):
        self.init = 0

    def generateSin(self, freq, amplitude, time, fs):
        n = time * fs
        x = np.linspace(0.0, time, n)
        s = amplitude * np.sin(freq * x * 2 * np.pi)
        return (x, s)

    def calcFFT(self, signal, fs):
        N = len(signal)
        T = 1 / fs
        xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), N)
        yf = fft(signal)
        return (xf, fftshift(yf))

    def plotFFT(self, signal, fs):
        x, y = self.calcFFT(signal, fs)
        plt.figure()
        plt.plot(x, np.abs(y))
        plt.title("Fourier")

    def play_sound(self, signal, fs):
        sd.default.samplerate = fs
        sd.default.channels = 1
        sd.play(signal)
        sd.wait()

    def read(self, reada):
        audio, samplerate = sf.read(reada)
        return audio, samplerate
