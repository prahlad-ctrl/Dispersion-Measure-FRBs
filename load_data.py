import torch
import h5py
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize
from synthetic_frbs import gen_synthetic_frb


class FRBDataset(Dataset):
    def __init__(self, mode='synthetic', real_data_dir='data/real_frbs', num_synthetic=5000):
        self.mode = mode
        self.num_synthetic = num_synthetic
        self.real_files = []

        if self.mode == 'real':
            if os.path.exists(real_data_dir):
                self.real_files = glob.glob(
                    os.path.join(real_data_dir, "*.h5"))

            # we should switch to synthetic if no real data found else would crash prolly
            if len(self.real_files) == 0:
                self.mode = 'synthetic'
                self.num_synthetic = 100

    def __len__(self):
        if self.mode == 'real':
            return len(self.real_files)
        else:
            return self.num_synthetic

    def redisperse_waterfall(self, waterfall, dm, freq_array, time_res_ms):
        k_dm = 4.148808
        f_high = freq_array[-1] / 1000.0
        freqs_ghz = freq_array / 1000.0
        delays_ms = k_dm * dm * (freqs_ghz**-2 - f_high**-2)
        delays_bins = (delays_ms / time_res_ms).astype(int)

        dispersed_wfall = np.zeros_like(waterfall)
        for i in range(waterfall.shape[0]):
            shift = delays_bins[i]
            if shift < waterfall.shape[1]:
                dispersed_wfall[i, :] = np.roll(waterfall[i, :], shift)
                if shift > 0:
                    dispersed_wfall[i, :shift] = np.random.normal(
                        0, 1, shift)  # add noise to prevent AI looking back
        return dispersed_wfall

    def __getitem__(self, idx):
        if self.mode == 'synthetic':
            dm = np.random.uniform(100, 2000)
            img = gen_synthetic_frb(dm)
            return torch.from_numpy(img).float().unsqueeze(0), torch.tensor([dm / 1000.0]).float()

        try:
            with h5py.File(self.real_files[idx], 'r') as f:
                dm, freq = f['frb'].attrs['dm'], f['frb']['plot_freq'][:]
                img = resize(np.nan_to_num(
                    f['frb']['wfall'][:]), (256, 256), anti_aliasing=True)

                freq_new = np.linspace(freq[0], freq[-1], 256)
                img = self.redisperse_waterfall(img, dm, freq_new, 1.0)
                img = (img - np.mean(img)) / (np.std(img) + 1e-6)

                return torch.from_numpy(img).float().unsqueeze(0), torch.tensor([dm / 1000.0]).float()
        except Exception:
            return self.__getitem__(0)