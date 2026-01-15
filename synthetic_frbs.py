import numpy as np


def gen_synthetic_frb(dm, n_freq=256, n_time=256, freq_low=400, freq_high=800):
    freqs = np.linspace(freq_high, freq_low, n_freq)

    k_dm = 4.1488e3

    # calculates delay for every single frequency channel (Cold Plasma Dispersion Relation)
    delays_ms = k_dm * dm * ((freqs)**-2 - (freq_high)**-2)

    delay_indices = (delays_ms / 1000.0 * n_time).astype(int)

    # bg noise like thermal, sky, etc.
    waterfall = np.random.normal(0, 1, (n_freq, n_time))

    pulse_width = 5
    center_time_bin = int(n_time * 0.2)

    for i in range(n_freq):
        t_start = center_time_bin + delay_indices[i]

        if t_start + pulse_width < n_time:
            signal_strength = np.random.uniform(5, 15)
            waterfall[i, t_start: t_start+pulse_width] += signal_strength

    # just so it is advantageous to NN dont think it matters much ig.
    waterfall = (waterfall - np.mean(waterfall)) / (np.std(waterfall) + 1e-6)

    return waterfall
