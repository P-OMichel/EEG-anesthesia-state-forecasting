import os
import json
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


CONDITIONS = ["ok", "shallow", "awake", "artifacted", "heavy", "too heavy"]
CONDITION_TO_ID = {name: idx for idx, name in enumerate(CONDITIONS)}
ID_TO_CONDITION = {idx: name for idx, name in enumerate(CONDITIONS)}


@dataclass
class EventMarker:
    time: float
    event_type: str  # "shallow_entry" or "bs_entry"


class AnnotationStore:
    """Holds ground-truth labels, valid crop boundaries, events, and .npz export/reload."""

    def __init__(self, fs: int = 128, win_sec: float = 30.0, step_sec: float = 10.0):
        self.fs = fs
        self.win_sec = win_sec
        self.step_sec = step_sec

        self.sample_id: str = "uninitialized"
        self.source_path: str = ""
        self.eeg_track_name: str = ""
        self.eeg_data: np.ndarray = np.array([], dtype=np.float32)
        self.t_windows: np.ndarray = np.array([], dtype=np.float32)
        self.powers: np.ndarray = np.array([], dtype=np.float32)
        self.states: np.ndarray = np.array([], dtype=np.int32)
        self.conditions: np.ndarray = np.array([], dtype=np.int32)
        self.events: List[EventMarker] = []

        self.keep_start_sec: float = 0.0
        self.keep_end_sec: float = 0.0

    def initialize_record(
        self,
        sample_id: str,
        eeg_data: np.ndarray,
        t_windows: np.ndarray,
        powers: np.ndarray,
        initial_states: np.ndarray,
        source_path: str = "",
        eeg_track_name: str = "",
    ):
        """Resets and populates the store for a newly loaded raw recording."""
        self.sample_id = sample_id
        self.source_path = source_path
        self.eeg_track_name = eeg_track_name
        self.eeg_data = eeg_data.astype(np.float32)
        self.t_windows = t_windows.astype(np.float32)
        self.powers = powers.astype(np.float32)
        self.states = initial_states.astype(np.int32)
        self.conditions = np.zeros(len(t_windows), dtype=np.int32)
        self.events.clear()

        self.keep_start_sec = 0.0
        self.keep_end_sec = float(len(eeg_data)) / float(self.fs)

    @staticmethod
    def is_annotated_npz(file_path: str) -> bool:
        """Returns True if the given .npz file already contains saved annotations."""
        if not file_path.lower().endswith(".npz") or not os.path.exists(file_path):
            return False
        try:
            with np.load(file_path, allow_pickle=False) as data:
                required = {"eeg", "t_windows", "state_labels", "condition_labels", "metadata_json"}
                return required.issubset(set(data.files))
        except Exception:
            return False

    @staticmethod
    def get_default_annotated_path(raw_file_path: str) -> str:
        """Returns the expected sidecar annotated filename for a raw VitalDB file."""
        directory = os.path.dirname(raw_file_path)
        stem = os.path.splitext(os.path.basename(raw_file_path))[0]
        if stem.endswith("_annotated"):
            return raw_file_path
        return os.path.join(directory, f"{stem}_annotated.npz")

    def load_from_annotated_npz(self, annotated_path: str):
        """Restores full recording + previously saved annotations for continued editing."""
        with np.load(annotated_path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))

            self.sample_id = str(data["sample_id"])
            self.source_path = annotated_path
            self.eeg_track_name = meta.get("eeg_track_name", "BIS__EEG1_WAV_val")
            self.eeg_data = data["eeg"].astype(np.float32)
            self.t_windows = data["t_windows"].astype(np.float32)
            self.powers = data["power_30s_10s"].astype(np.float32)
            self.states = data["state_labels"].astype(np.int32)
            self.conditions = data["condition_labels"].astype(np.int32)

            max_t = float(len(self.eeg_data)) / float(self.fs)
            self.keep_start_sec = float(meta.get("crop_start_sec", 0.0))
            self.keep_end_sec = float(meta.get("crop_end_sec", max_t))

            self.events.clear()
            for t_sh in data.get("shallow_entry_times", []):
                self.events.append(EventMarker(time=float(t_sh), event_type="shallow_entry"))
            for t_bs in data.get("bs_entry_times", []):
                self.events.append(EventMarker(time=float(t_bs), event_type="bs_entry"))

    def set_crop_bounds(self, start_sec: float, end_sec: float):
        max_t = float(len(self.eeg_data)) / float(self.fs)
        self.keep_start_sec = max(0.0, min(float(start_sec), max_t))
        self.keep_end_sec = max(self.keep_start_sec, min(float(end_sec), max_t))

    def set_states_at_indices(self, indices: Sequence[int], values: Sequence[int]):
        for idx, val in zip(indices, values):
            if 0 <= idx < len(self.states):
                self.states[idx] = int(np.clip(round(val), 0, 20))

    def set_conditions_at_indices(self, indices: Sequence[int], condition_idx: int):
        if condition_idx not in ID_TO_CONDITION:
            raise ValueError(f"Invalid condition index: {condition_idx}")
        for idx in indices:
            if 0 <= idx < len(self.conditions):
                self.conditions[idx] = condition_idx

    def add_event(self, time_sec: float, event_type: str) -> EventMarker:
        if event_type not in ("shallow_entry", "bs_entry"):
            raise ValueError(f"Unsupported event type: {event_type}")
        marker = EventMarker(time=float(time_sec), event_type=event_type)
        self.events.append(marker)
        return marker

    def update_event_time(self, index: int, new_time_sec: float):
        if 0 <= index < len(self.events):
            self.events[index].time = float(new_time_sec)

    def remove_event(self, index: int):
        if 0 <= index < len(self.events):
            self.events.pop(index)

    def save_annotated_npz(self, save_path: str) -> dict:
        """
        Saves the full recording, valid crop bounds, aligned labels, and event timestamps
        to compressed .npz (excluding spectrogram).
        Saving full arrays + crop bounds allows lossless re-opening and boundary adjustment later.
        """
        valid_win_mask = (self.t_windows >= self.keep_start_sec) & (
            self.t_windows <= self.keep_end_sec
        )

        shallow_entries = np.array(
            [ev.time for ev in self.events if ev.event_type == "shallow_entry"],
            dtype=np.float32,
        )
        bs_entries = np.array(
            [ev.time for ev in self.events if ev.event_type == "bs_entry"],
            dtype=np.float32,
        )

        metadata = {
            "sample_id": self.sample_id,
            "eeg_track_name": self.eeg_track_name,
            "source_file": os.path.basename(self.source_path) if self.source_path else "",
            "sampling_rate_hz": self.fs,
            "window_size_sec": self.win_sec,
            "step_size_sec": self.step_sec,
            "crop_start_sec": round(self.keep_start_sec, 3),
            "crop_end_sec": round(self.keep_end_sec, 3),
            "condition_mapping": CONDITION_TO_ID,
        }

        np.savez_compressed(
            save_path,
            sample_id=np.array(self.sample_id),
            eeg=self.eeg_data,
            t_windows=self.t_windows,
            valid_window_mask=valid_win_mask,
            power_30s_10s=self.powers,
            state_labels=self.states,
            condition_labels=self.conditions,
            shallow_entry_times=shallow_entries,
            bs_entry_times=bs_entries,
            metadata_json=np.array(json.dumps(metadata)),
        )

        start_sample = int(round(self.keep_start_sec * self.fs))
        end_sample = int(round(self.keep_end_sec * self.fs))
        valid_events = sum(
            1 for ev in self.events if self.keep_start_sec <= ev.time <= self.keep_end_sec
        )

        return {
            "kept_samples": max(0, end_sample - start_sample),
            "kept_duration_sec": max(0.0, self.keep_end_sec - self.keep_start_sec),
            "kept_windows": int(np.sum(valid_win_mask)),
            "kept_events": valid_events,
        }

    @staticmethod
    def load_cropped_for_ml(annotated_path: str) -> dict:
        """
        Loads an annotated .npz file for supervised ML training, automatically applying
        the saved [crop_start_sec, crop_end_sec] boundaries and shifting timestamps to t=0.
        """
        with np.load(annotated_path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            fs = int(meta["sampling_rate_hz"])
            crop_start = float(meta["crop_start_sec"])
            crop_end = float(meta["crop_end_sec"])

            s0 = int(round(crop_start * fs))
            s1 = int(round(crop_end * fs))
            eeg_cropped = data["eeg"][s0:s1]

            mask = data["valid_window_mask"].astype(bool)
            t_win_cropped = data["t_windows"][mask] - crop_start
            states_cropped = data["state_labels"][mask]
            conds_cropped = data["condition_labels"][mask]

            sh = data["shallow_entry_times"]
            bs = data["bs_entry_times"]
            sh_cropped = sh[(sh >= crop_start) & (sh <= crop_end)] - crop_start
            bs_cropped = bs[(bs >= crop_start) & (bs <= crop_end)] - crop_start

            return {
                "sample_id": str(data["sample_id"]),
                "eeg": eeg_cropped,
                "t_windows": t_win_cropped,
                "state_labels": states_cropped,
                "condition_labels": conds_cropped,
                "shallow_entry_times": sh_cropped,
                "bs_entry_times": bs_cropped,
                "metadata": meta,
            }