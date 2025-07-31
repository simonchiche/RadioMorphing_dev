#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:34:36 2024

@author: chiche
"""


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Define the time domain signal E(t)
def E(t):
    # Example: a combination of two sine waves
    return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

# Define the time range
T = 1.0      # Total time period (in seconds)
N = 500      # Number of sample points
dt = T / N   # Sampling interval
t = np.linspace(0, T, N, endpoint=False)  # Time array

# Compute the signal E(t)
signal = E(t)

# Compute the Fourier Transform using FFT
fft_result = np.fft.fft(signal)
fft_freqs = np.fft.fftfreq(N, dt)

# Compute the magnitude of the FFT result
fft_magnitude = np.abs(fft_result) / N

# Since FFT output is symmetrical, take only the positive half
positive_freqs = fft_freqs[:N // 2]
positive_magnitude = fft_magnitude[:N // 2]

# Plot the Fourier Transform of E(t)
plt.figure(figsize=(10, 5))
plt.plot(positive_freqs, positive_magnitude)
plt.title('Fourier Transform of E(t)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.show()
