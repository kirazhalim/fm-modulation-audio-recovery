import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import soundfile as sf

def ciz_spectrum(sinyal, fs, baslik, sinir=None):
    spektrum = np.fft.fftshift(np.fft.fft(sinyal))
    frekanslar = np.fft.fftshift(np.fft.fftfreq(len(sinyal), 1/fs))
    plt.figure()
    plt.plot(frekanslar/1e3, np.abs(spektrum))
    plt.xlabel("Frekans (kHz)")
    plt.ylabel("Genlik")
    plt.title(baslik)
    if sinir:
        plt.xlim(-sinir/1e3, sinir/1e3)
    plt.grid()
    plt.show()

data = sio.loadmat("fm_signal.mat")
fm_sinyali = data["band_pass_signal"].squeeze()
fs = int(data["fs"].squeeze())
fc = float(data["fc"].squeeze())

ciz_spectrum(fm_sinyali, fs, "Giris FM Spektrumu", sinir=120000)

analitik_sinyal = signal.hilbert(fm_sinyali)
faz = np.unwrap(np.angle(analitik_sinyal))
anlik_frekans = np.diff(faz) * fs / (2 * np.pi)

anlik_frekans -= np.mean(anlik_frekans)
anlik_frekans /= np.max(np.abs(anlik_frekans))

ciz_spectrum(anlik_frekans, fs, "Demodüle Edilen Sinyal", sinir=60000)

b, a = signal.butter(8, 15000 / (fs/2), btype="low")
mono_sinyal = signal.filtfilt(b, a, anlik_frekans)

tau = 75e-6
b_de, a_de = signal.bilinear([1], [tau, 1], fs)
mono_filtreli = signal.filtfilt(b_de, a_de, mono_sinyal)

yeni_fs = 48000
cikis_sesi = signal.resample_poly(mono_filtreli, yeni_fs, fs)

cikis_sesi = cikis_sesi / (np.std(cikis_sesi) * 3)
cikis_sesi = np.clip(cikis_sesi, -1.0, 1.0)

ciz_spectrum(cikis_sesi, yeni_fs, "Son Ses Spektrumu", sinir=20000)

sf.write("output.wav", cikis_sesi, yeni_fs)