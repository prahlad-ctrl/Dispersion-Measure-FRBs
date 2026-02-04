import torch
import h5py
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize
from synthetic_frbs import gen_synthetic_frb


class FRBDataset(Dataset):
    def __init__(self, mode= 'synthetic', real_data_dir='data/real_frbs', num_synthetic= 5000):
        self.mode = mode
        self.num_synthetic = num_synthetic
        self.real_files = []

        if self.mode == 'real':
            if os.path.exists(real_data_dir):
                self.real_files = glob.glob(
                    os.path.join(real_data_dir, "*.h5"))

            if len(self.real_files) == 0:
                self.mode = 'synthetic'
                self.num_synthetic = 100

    def __len__(self):
        return len(self.real_files) if self.mode == 'real' else self.num_synthetic

    # Explained why redispersion is needed in raw/read.txt
    def redisperse_waterfall(self, waterfall, dm, freq_array, time_res_ms):
        k_dm = 4.1488e3
        f_high = freq_array[-1]

        delays_ms = k_dm * dm * (freq_array**-2 - f_high**-2) # Dispersion delay equation
        delays_bins = (delays_ms / time_res_ms).astype(int)

        dispersed_wfall = np.zeros_like(waterfall)
        for i in range(waterfall.shape[0]):
            shift = delays_bins[i]
            if shift < waterfall.shape[1]:
                dispersed_wfall[i, :] = np.roll(waterfall[i, :], shift) # applies dispersion delay by shifting signal
                if shift > 0:
                    dispersed_wfall[i, :shift] = np.random.normal(0, 1, shift) # fill element if shifted = true
            else:
                dispersed_wfall[i, :] = np.random.normal(
                    0, 1, waterfall.shape[1]) # fills entire frequency channel with noise (signal completely shifted out)
        return dispersed_wfall

    def __getitem__(self, idx):
        if self.mode == 'synthetic':
            dm = np.random.uniform(100, 2000) # could try diff values but this range is reasonable
            img = gen_synthetic_frb(dm)
            return torch.from_numpy(img).float().unsqueeze(0), torch.tensor([dm / 1000.0]).float() # converting to tensor, normalizing dm

        try:
            with h5py.File(self.real_files[idx], 'r') as f:
                dm = f['frb'].attrs['dm']
                raw_data = np.nan_to_num(f['frb']['wfall'][:])
                img = resize(raw_data, (256, 256), anti_aliasing=True)

                freq_new = np.linspace(800, 400, 256)
                synthetic_time_res = 1000.0 / 256.0 # same roughly 3.6ms per pixel
                img = self.redisperse_waterfall(
                    img, dm, freq_new, synthetic_time_res)

                ' robust normalization to handle RFI and outliers '
                median = np.median(img)
                q75, q25 = np.percentile(img, [75, 25])
                iqr = q75 - q25 # interquartile range
                sigma_robust = iqr / 1.3489

                if sigma_robust == 0:
                    sigma_robust = 1e-6

                ' explained in synthetic_frbs.py '
                img = np.clip(img, median - 3*sigma_robust,
                              median + 3*sigma_robust)
                img = (img - median) / sigma_robust

                return torch.from_numpy(img).float().unsqueeze(0), torch.tensor([dm / 1000.0]).float()
        except Exception as e:
            return self.__getitem__(0)
