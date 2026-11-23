import os
import numpy as np
from scipy import signal


class EEGProcessor:
    """Handles VitalDB EEG extraction, spectral analysis, and state estimation."""

    PREFERRED_EEG_KEYS = ["BIS__EEG1_WAV_val", "BIS__EEG2_WAV_val"]

    def __init__(self, fs: int = 128):
        self.fs = fs

    def extract_eeg_from_file(self, file_path: str) -> tuple[np.ndarray, str]:
        """
        Loads 1D EEG signal from either:
        - A raw VitalDB exported .npz ('BIS__EEG1_WAV_val' or 'BIS__EEG2_WAV_val')
        - An already-annotated .npz ('eeg')
        - A raw 1D .npy file
        Returns (clean_eeg_1d, track_name_used).
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".npy":
            arr = np.load(file_path).squeeze()
            if arr.ndim > 1:
                arr = arr[0]
            return self._clean_nan(arr), "npy_channel_0"

        if ext == ".npz":
            with np.load(file_path, allow_pickle=False) as data:
                # 1. Check if it is an already-annotated file
                if "eeg" in data.files:
                    return self._clean_nan(data["eeg"]), "annotated_eeg"

                # 2. Check VitalDB exported EEG keys
                for key in self.PREFERRED_EEG_KEYS:
                    if key in data.files:
                        return self._clean_nan(data[key]), key

                raise KeyError(
                    f"No EEG waveform found in {os.path.basename(file_path)}. "
                    f"Available keys: {data.files}"
                )

        raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _clean_nan(arr: np.ndarray) -> np.ndarray:
        """Converts NaN/Inf segments (e.g. unplugged leads in VitalDB) to 0.0."""
        arr = np.asarray(arr, dtype=np.float32).squeeze()
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_spectrogram(self, eeg: np.ndarray):
        """Computes STFT spectrogram for display (excluded from saved files)."""
        nperseg = self.fs * 4      # 4-second windows
        noverlap = self.fs * 3     # 3-second overlap
        f, t_spec, Sxx = signal.spectrogram(
            eeg, fs=self.fs, nperseg=nperseg, noverlap=noverlap
        )
        freq_mask = (f >= 0.5) & (f <= 45.0)
        f = f[freq_mask]
        Sxx_db = 10 * np.log10(Sxx[freq_mask, :] + 1e-12)
        return f, t_spec, Sxx_db

    def compute_sliding_power(
        self, eeg: np.ndarray, win_sec: float = 30.0, step_sec: float = 10.0
    ):
        """
        Computes mean power over sliding windows of 30s taken every 10s.
        Returns window center timestamps and power array.
        """
        win_samples = int(win_sec * self.fs)
        step_samples = int(step_sec * self.fs)

        if len(eeg) < win_samples:
            raise ValueError("EEG recording is shorter than the 30s analysis window.")

        starts = np.arange(0, len(eeg) - win_samples + 1, step_samples)
        t_windows = (starts + win_samples / 2.0) / self.fs
        powers = np.zeros(len(starts), dtype=np.float32)

        for i, start in enumerate(starts):
            segment = eeg[start : start + win_samples]
            powers[i] = np.mean(segment ** 2)

        return t_windows, powers

    def estimate_initial_state(self, powers: np.ndarray) -> np.ndarray:
        """
        PLACEHOLDER ALGORITHM: Maps power levels to integer states [0..20] in steps of 1.
        High power -> 20, Low power -> 0.
        Replace this method later with your custom algorithm.
        """
        # Compute percentiles on non-zero power windows to ignore unplugged lead zeros
        valid_powers = powers[powers > 0]
        if len(valid_powers) == 0:
            return np.zeros(len(powers), dtype=np.int32)

        p_low = np.percentile(valid_powers, 5)
        p_high = np.percentile(valid_powers, 95)

        if p_high <= p_low:
            return np.full(len(powers), 10, dtype=np.int32)

        normalized = (powers - p_low) / (p_high - p_low)
        clipped = np.clip(normalized, 0.0, 1.0)
        return np.round(clipped * 20.0).astype(np.int32)