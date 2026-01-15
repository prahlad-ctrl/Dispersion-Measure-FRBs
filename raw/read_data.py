import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# just took one example file to show visualization but its de-dispersed so just straight lines ig
file_path = os.path.join('..', 'data', 'real_frbs',
                         'FRB20180814A_waterfall.h5')

with h5py.File(file_path, 'r') as f:
    data = f['frb']['wfall'][:]

    freqs = f['frb']['plot_freq'][:]
    times = f['frb']['plot_time'][:]

    extent = [times[0], times[-1], freqs[0], freqs[-1]]

    plt.figure(figsize=(10, 6))
    plt.imshow(data, aspect='auto', origin='lower',
               cmap='inferno', extent=extent)

    plt.colorbar(label='Flux Density / Intensity')
    plt.xlabel('Time (ms)')
    plt.ylabel('Frequency (MHz)')
    plt.title('FRB Waterfall (Calibrated)')
    plt.savefig('frb_waterfall.png', dpi=300)
    plt.show()
