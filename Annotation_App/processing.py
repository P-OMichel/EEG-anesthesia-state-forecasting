import os
import numpy as np
from scipy import signal

# Update these two import paths to match your project structure
from Annotation_App.State_Chart.compute_variables import ComputeVariables
from Annotation_App.State_Chart.state_chart import EstimateState


class EEGProcessor:
    """Handles VitalDB EEG & hypnotic extraction, spectral analysis, and state estimation."""

    PREFERRED_EEG_KEYS = ["BIS__EEG1_WAV_val", "BIS__EEG2_WAV_val"]

    PROPOFOL_TRACKS = {
        "Orchestra__PPF20_CE": "Propofol Ce (µg/mL)",
        "Orchestra__PPF20_CP": "Propofol Cp (µg/mL)",
        "Orchestra__PPF_CE": "Propofol Ce (µg/mL)",
        "Orchestra__PPF_CP": "Propofol Cp (µg/mL)",
    }

    SEVO_TRACKS = {
        "Primus__EXP_SEVO": "Sevoflurane Exp (%)",
        "Solar8000__ETSEVO": "Sevoflurane ET (%)",
        "Primus__INSP_SEVO": "Sevoflurane Insp (%)",
        "Solar8000__INSEVO": "Sevoflurane Insp (%)",
    }

    FEATURE_KEYS = [
        "prop_delta",
        "prop_alpha",
        "prop_beta",
        "prop_gamma",
        "ies_prop",
        "alpha_supp_prop",
        "r_alpha_delta",
        "r_beta_delta",
        "r_gamma_delta",
        "r_beta_alpha",
        "r_gamma_alpha",
        "r_gamma_beta",
        "q_0_85",
        "f_central",
    ]

    def __init__(self, fs: int = 128):
        self.fs = fs
        self.var_extractor = ComputeVariables(fs=self.fs)
        self.state_estimator = EstimateState()

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
                if "eeg" in data.files:
                    return self._clean_nan(data["eeg"]), "annotated_eeg"

                for key in self.PREFERRED_EEG_KEYS:
                    if key in data.files:
                        return self._clean_nan(data[key]), key

                raise KeyError(
                    f"No EEG waveform found in {os.path.basename(file_path)}. "
                    f"Available keys: {data.files}"
                )

        raise ValueError(f"Unsupported file format: {ext}")

    def extract_hypnotics_from_file(self, file_path: str) -> tuple[list[dict], str]:
        """
        Scans a VitalDB .npz file for Propofol or Sevoflurane concentration tracks.
        Returns:
            tracks: list of dicts {'name': str, 'agent': str, 'time': np.ndarray, 'val': np.ndarray}
            detected_agent: 'propofol' | 'sevoflurane' | 'mixed' | 'unknown'
        """
        tracks = []
        if not file_path.lower().endswith(".npz") or not os.path.exists(file_path):
            return tracks, "unknown"

        has_propofol = False
        has_sevo = False

        with np.load(file_path, allow_pickle=False) as data:
            file_keys = set(data.files)

            for base_key, display_name in self.PROPOFOL_TRACKS.items():
                val_key = f"{base_key}_val"
                time_key = f"{base_key}_time"
                if val_key in file_keys and time_key in file_keys:
                    vals = np.asarray(data[val_key], dtype=np.float32)
                    times = np.asarray(data[time_key], dtype=np.float32)
                    valid = ~np.isnan(vals)
                    if np.any(valid) and np.nanmax(vals[valid]) > 0.0:
                        has_propofol = True
                        tracks.append({
                            "name": display_name,
                            "agent": "propofol",
                            "time": times[valid],
                            "val": vals[valid],
                        })

            for base_key, display_name in self.SEVO_TRACKS.items():
                val_key = f"{base_key}_val"
                time_key = f"{base_key}_time"
                if val_key in file_keys and time_key in file_keys:
                    vals = np.asarray(data[val_key], dtype=np.float32)
                    times = np.asarray(data[time_key], dtype=np.float32)
                    valid = ~np.isnan(vals)
                    if np.any(valid) and np.nanmax(vals[valid]) > 0.0:
                        has_sevo = True
                        tracks.append({
                            "name": display_name,
                            "agent": "sevoflurane",
                            "time": times[valid],
                            "val": vals[valid],
                        })

        if has_propofol and has_sevo:
            detected_agent = "mixed"
        elif has_propofol:
            detected_agent = "propofol"
        elif has_sevo:
            detected_agent = "sevoflurane"
        else:
            detected_agent = "unknown"

        return tracks, detected_agent

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

    def _estimate_window_state_and_features(self, segment: np.ndarray) -> tuple[int, dict]:
        """Runs variable extraction and state estimation on a single EEG window."""
        if np.all(segment == 0):
            return 0, {k: 0.0 for k in self.FEATURE_KEYS}

        cv = self.var_extractor
        cv.clean_signal(segment)
        cv.get_P_list()
        cv.get_P_prop()
        cv.get_P_ratios()
        cv.get_f_main()
        cv.get_f_quantiles()
        cv.get_ef()
        cv.get_supp()

        feats = {
            "prop_delta": float(cv.prop_delta),
            "prop_alpha": float(cv.prop_alpha),
            "prop_beta": float(cv.prop_beta),
            "prop_gamma": float(cv.prop_gamma),
            "ies_prop": float(cv.ies_prop),
            "alpha_supp_prop": float(cv.alpha_supp_prop),
            "r_alpha_delta": float(cv.r_alpha_delta),
            "r_beta_delta": float(cv.r_beta_delta),
            "r_gamma_delta": float(cv.r_gamma_delta),
            "r_beta_alpha": float(cv.r_beta_alpha),
            "r_gamma_alpha": float(cv.r_gamma_alpha),
            "r_gamma_beta": float(cv.r_gamma_beta),
            "q_0_85": float(cv.q_0_85),
            "f_central": float(cv.f_central),
        }

        self.state_estimator.current_data(
            prop_delta=cv.prop_delta,
            prop_alpha=cv.prop_alpha,
            prop_beta=cv.prop_beta,
            prop_gamma=cv.prop_gamma,
            r_alpha_delta=cv.r_alpha_delta,
            r_beta_delta=cv.r_beta_delta,
            r_gamma_delta=cv.r_gamma_delta,
            r_beta_alpha=cv.r_beta_alpha,
            r_gamma_alpha=cv.r_gamma_alpha,
            r_gamma_beta=cv.r_gamma_beta,
            q_0_85=cv.q_0_85,
            f_central=cv.f_central,
            ies_prop=cv.ies_prop,
            alpha_supp_prop=cv.alpha_supp_prop,
        )
        self.state_estimator.get_main_state()
        self.state_estimator.get_state()
        state = int(self.state_estimator.state)

        return state, feats

    def compute_sliding_states(
        self, eeg: np.ndarray, win_sec: float = 30.0, step_sec: float = 10.0
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """
        Computes the integer state [0..20] and all intermediate features over sliding
        windows of 30s taken every 10s.
        Returns (t_windows, states, features_dict).
        """
        win_samples = int(win_sec * self.fs)
        step_samples = int(step_sec * self.fs)

        if len(eeg) < win_samples:
            raise ValueError("EEG recording is shorter than the 30s analysis window.")

        starts = np.arange(0, len(eeg) - win_samples + 1, step_samples)
        n_win = len(starts)
        t_windows = (starts + win_samples / 2.0) / self.fs
        states = np.zeros(n_win, dtype=np.int32)
        features = {k: np.zeros(n_win, dtype=np.float32) for k in self.FEATURE_KEYS}

        for i, start in enumerate(starts):
            segment = eeg[start : start + win_samples]
            st, w_feats = self._estimate_window_state_and_features(segment)
            states[i] = st
            for k in self.FEATURE_KEYS:
                features[k][i] = w_feats[k]

        return t_windows, states, features