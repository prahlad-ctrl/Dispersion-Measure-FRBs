import numpy as np
import scipy.ndimage as ndimage

'''
Synthetic FRBs data is needed for testing because real FRB data is not always available.
This module generates synthetic FRB waterfall data with realistic characteristics, including dispersion, noise, and signal profiles.
We can also say that there is 'shortage of data' bacause real FRB events are rare and limited in number
'''

'''
CHIME (leading frb detectors) operates in the frequency range of 400 MHz to 800 MHz.
'''
def gen_synthetic_frb(dm, n_freq=256, n_time=256, freq_low=400, freq_high=800):

    freqs = np.linspace(freq_high, freq_low, 256)
    k_dm = 4.1488e3  # Dispersion constant in MHz^2 pc^-1 cm^3 ms

    ' Calculating dispersion delay of signals in milliseconds '
    delays_ms = k_dm * dm * ((freqs)**-2 - (freq_high)**-2) # Dispersion delay equation used for radio waves travelling through ISM
    time_res = 1000.0 / n_time # each pixel represents roughly 3.9 ms 
    delay_indices = (delays_ms / time_res).astype(int) # converting delay from ms to array indices

    ' Adding random noise and data to match realistic FRB characteristics '
    waterfall = np.random.normal(0, 2.0, (n_freq, n_time))

    channel_gains = np.random.normal(0, 5.0, (n_freq, 1))
    waterfall += channel_gains

    blobs = ndimage.gaussian_filter(
        np.random.normal(0, 2.0, (n_freq, n_time)), sigma=2.0)
    waterfall += blobs # stimulates RFI and atmospheric effects

    ' Randomize even these features so they dont overfit to a specific pattern '
    center_time_bin = np.random.randint(int(n_time * 0.15), int(n_time * 0.7)) # ensures signal is visible and not cutoff at edges
    pulse_width = np.random.randint(10, 40)

    ' Creating the profile of FRB signal '
    for i in range(n_freq):
        t_start = center_time_bin + delay_indices[i]
        if t_start + pulse_width < n_time:
            x = np.linspace(-2, 2, pulse_width)
            profile = np.exp(-x**2) # bell curve for realistic appearance

            signal_strength = np.random.uniform(5, 12)
            waterfall[i, t_start: t_start +
                      pulse_width] += profile * signal_strength

    waterfall = ndimage.gaussian_filter(waterfall, sigma=(0.5, 1.0))

    '''
    In radio astronomy, a single massive interference spike can ruin the scale of the whole image.
    This step ensures that the extreme outliers dont wash out the rest of the image features, 
    keeping the background texture visible.
    '''
    
    ' Calculates std and trims the image to remove extreme outliers '
    std_val = np.std(waterfall)
    waterfall = np.clip(waterfall, -3*std_val, 3*std_val)

    waterfall = (waterfall - np.mean(waterfall)) / (np.std(waterfall) + 1e-6) # basic z-score normalization

    return waterfall
