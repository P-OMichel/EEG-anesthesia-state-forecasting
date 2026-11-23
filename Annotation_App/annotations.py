import os
import json
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np


CONDITIONS = ["ok", "shallow", "awake", "artifacted", "heavy", "too heavy"]
CONDITION_TO_ID = {name: idx for idx, name in enumerate(CONDITIONS)}
ID_TO_CONDITION = {idx: name for idx, name in enumerate(CONDITIONS)}

HYPNOTIC_AGENTS = ["unknown", "propofol", "sevoflurane", "mixed"]


@dataclass
class EventMarker:
    time: float
    event_type: str  # "shallow_entry" or "bs_entry"


class AnnotationStore:
    """Holds ground-truth labels, hypnotic agent info, crop boundaries, and selective .npz I/O."""

    def __init__(self, fs: int = 128, win_sec: float = 30.0, step_sec: float = 10.0):
        self.fs = fs
        self.win_sec = win_sec
        self.step_sec = step_sec

        self.sample_id: str = "uninitialized"
        self.source_path: str = ""
        self.eeg_track_name: str = ""
        self.hypnotic_type: str = "unknown"
        self.hypnotic_tracks: List[dict] = []

        self.eeg_data: np.ndarray = np.array([], dtype=np.float32)
        self.t_windows: np.ndarray = np.array([], dtype=np.float32)
        self.powers: np.ndarray = np.array([], dtype=np.float32)
        self.states: np.ndarray = np.array([], dtype=np.int32)
        self.conditions: np.ndarray = np.array([], dtype=np.int32)
        self.events: List[EventMarker] = []

        self.keep_start_sec: float = 0.0
        self.keep_end_sec: float = 0.0

        # Tracks which sections were present when loading an existing annotated file
        self.loaded_sections: dict = {
            "states": True,
            "conditions": True,
            "events": True,
            "crop": True,
        }

    def initialize_record(
        self,
        sample_id: str,
        eeg_data: np.ndarray,
        t_windows: np.ndarray,
        powers: np.ndarray,
        initial_states: np.ndarray,
        source_path: str = "",
        eeg_track_name: str = "",
        hypnotic_type: str = "unknown",
        hypnotic_tracks: List[dict] | None = None,
    ):
        """Resets and populates the store for a newly loaded raw recording."""
        self.sample_id = sample_id
        self.source_path = source_path
        self.eeg_track_name = eeg_track_name
        self.hypnotic_type = hypnotic_type if hypnotic_type in HYPNOTIC_AGENTS else "unknown"
        self.hypnotic_tracks = hypnotic_tracks if hypnotic_tracks is not None else []

        self.eeg_data = eeg_data.astype(np.float32)
        self.t_windows = t_windows.astype(np.float32)
        self.powers = powers.astype(np.float32)
        self.states = initial_states.astype(np.int32)
        self.conditions = np.zeros(len(t_windows), dtype=np.int32)
        self.events.clear()

        self.keep_start_sec = 0.0
        self.keep_end_sec = float(len(eeg_data)) / float(self.fs)
        self.loaded_sections = {"states": True, "conditions": True, "events": True, "crop": True}

    @staticmethod
    def is_annotated_npz(file_path: str) -> bool:
        """Returns True if the given .npz file is an annotated archive created by this tool."""
        if not file_path.lower().endswith(".npz") or not os.path.exists(file_path):
            return False
        try:
            with np.load(file_path, allow_pickle=False) as data:
                # Core base keys always written when saving an annotated file
                base_required = {"sample_id", "eeg", "t_windows", "metadata_json"}
                return base_required.issubset(set(data.files))
        except Exception:
            return False

    @staticmethod
    def get_default_annotated_path(raw_file_path: str) -> str:
        """Returns the expected sidecar annotated filename in the exact same folder."""
        directory = os.path.dirname(raw_file_path)
        stem = os.path.splitext(os.path.basename(raw_file_path))[0]
        if stem.endswith("_annotated"):
            return raw_file_path
        return os.path.join(directory, f"{stem}_annotated.npz")

    def load_from_annotated_npz(
        self,
        annotated_path: str,
        fallback_initial_states_fn=None,
        hypnotic_tracks: List[dict] | None = None,
    ):
        """
        Restores recording and any selectively saved annotation sections.
        Sections not saved in the archive fall back cleanly to defaults/initial estimates.
        """
        with np.load(annotated_path, allow_pickle=False) as data:
            file_keys = set(data.files)
            meta = json.loads(str(data["metadata_json"]))
            saved_sections = meta.get("saved_sections", {})

            self.sample_id = str(data["sample_id"])
            self.source_path = annotated_path
            self.eeg_track_name = meta.get("eeg_track_name", "BIS__EEG1_WAV_val")
            self.hypnotic_type = (
                str(data["hypnotic_type"])
                if "hypnotic_type" in file_keys
                else meta.get("hypnotic_type", "unknown")
            )
            self.hypnotic_tracks = hypnotic_tracks if hypnotic_tracks is not None else []

            self.eeg_data = data["eeg"].astype(np.float32)
            self.t_windows = data["t_windows"].astype(np.float32)
            self.powers = data["power_30s_10s"].astype(np.float32)

            # 1. States
            has_states = saved_sections.get("states", "state_labels" in file_keys) and (
                "state_labels" in file_keys
            )
            if has_states:
                self.states = data["state_labels"].astype(np.int32)
            elif fallback_initial_states_fn is not None:
                self.states = fallback_initial_states_fn(self.powers).astype(np.int32)
            else:
                self.states = np.zeros(len(self.t_windows), dtype=np.int32)

            # 2. Clinical Conditions
            has_conditions = saved_sections.get(
                "conditions", "condition_labels" in file_keys
            ) and ("condition_labels" in file_keys)
            if has_conditions:
                self.conditions = data["condition_labels"].astype(np.int32)
            else:
                self.conditions = np.zeros(len(self.t_windows), dtype=np.int32)

            # 3. Crop Boundaries
            max_t = float(len(self.eeg_data)) / float(self.fs)
            has_crop = saved_sections.get("crop", "crop_start_sec" in meta)
            if has_crop:
                self.keep_start_sec = float(meta.get("crop_start_sec", 0.0))
                self.keep_end_sec = float(meta.get("crop_end_sec", max_t))
            else:
                self.keep_start_sec = 0.0
                self.keep_end_sec = max_t

            # 4. Entry Events
            has_events = saved_sections.get(
                "events",
                ("shallow_entry_times" in file_keys) or ("bs_entry_times" in file_keys),
            )
            self.events.clear()
            if has_events:
                for t_sh in data.get("shallow_entry_times", []):
                    self.events.append(EventMarker(time=float(t_sh), event_type="shallow_entry"))
                for t_bs in data.get("bs_entry_times", []):
                    self.events.append(EventMarker(time=float(t_bs), event_type="bs_entry"))

            self.loaded_sections = {
                "states": bool(has_states),
                "conditions": bool(has_conditions),
                "events": bool(has_events),
                "crop": bool(has_crop),
            }

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

    def save_annotated_npz(
        self,
        save_path: str,
        save_states: bool = True,
        save_conditions: bool = True,
        save_events: bool = True,
        save_crop: bool = True,
    ) -> dict:
        """
        Selectively saves chosen annotation tracks to compressed .npz (excluding spectrogram).
        Unselected sections are omitted so partial annotations never pollute ML datasets.
        """
        max_t = float(len(self.eeg_data)) / float(self.fs)
        effective_start = self.keep_start_sec if save_crop else 0.0
        effective_end = self.keep_end_sec if save_crop else max_t

        valid_win_mask = (self.t_windows >= effective_start) & (
            self.t_windows <= effective_end
        )

        saved_sections = {
            "states": bool(save_states),
            "conditions": bool(save_conditions),
            "events": bool(save_events),
            "crop": bool(save_crop),
        }

        metadata = {
            "sample_id": self.sample_id,
            "eeg_track_name": self.eeg_track_name,
            "hypnotic_type": self.hypnotic_type,
            "source_file": os.path.basename(self.source_path) if self.source_path else "",
            "sampling_rate_hz": self.fs,
            "window_size_sec": self.win_sec,
            "step_size_sec": self.step_sec,
            "crop_start_sec": round(effective_start, 3),
            "crop_end_sec": round(effective_end, 3),
            "condition_mapping": CONDITION_TO_ID,
            "saved_sections": saved_sections,
        }

        payload = {
            "sample_id": np.array(self.sample_id),
            "hypnotic_type": np.array(self.hypnotic_type),
            "eeg": self.eeg_data,
            "t_windows": self.t_windows,
            "valid_window_mask": valid_win_mask,
            "power_30s_10s": self.powers,
            "metadata_json": np.array(json.dumps(metadata)),
        }

        if save_states:
            payload["state_labels"] = self.states

        if save_conditions:
            payload["condition_labels"] = self.conditions

        kept_events_count = 0
        if save_events:
            shallow_entries = np.array(
                [ev.time for ev in self.events if ev.event_type == "shallow_entry"],
                dtype=np.float32,
            )
            bs_entries = np.array(
                [ev.time for ev in self.events if ev.event_type == "bs_entry"],
                dtype=np.float32,
            )
            payload["shallow_entry_times"] = shallow_entries
            payload["bs_entry_times"] = bs_entries
            kept_events_count = sum(
                1 for ev in self.events if effective_start <= ev.time <= effective_end
            )

        for idx, trk in enumerate(self.hypnotic_tracks):
            payload[f"hypnotic_{idx}_time"] = trk["time"]
            payload[f"hypnotic_{idx}_val"] = trk["val"]
            payload[f"hypnotic_{idx}_name"] = np.array(trk["name"])

        np.savez_compressed(save_path, **payload)
        self.loaded_sections = saved_sections

        start_sample = int(round(effective_start * self.fs))
        end_sample = int(round(effective_end * self.fs))

        return {
            "kept_samples": max(0, end_sample - start_sample),
            "kept_duration_sec": max(0.0, effective_end - effective_start),
            "kept_windows": int(np.sum(valid_win_mask)),
            "kept_events": kept_events_count,
            "saved_sections": saved_sections,
        }

    @staticmethod
    def load_cropped_for_ml(annotated_path: str) -> dict:
        """
        Loads an annotated .npz file for supervised ML training.
        Sections that were unchecked during saving are returned as None.
        """
        with np.load(annotated_path, allow_pickle=False) as data:
            file_keys = set(data.files)
            meta = json.loads(str(data["metadata_json"]))
            fs = int(meta["sampling_rate_hz"])
            crop_start = float(meta["crop_start_sec"])
            crop_end = float(meta["crop_end_sec"])

            s0 = int(round(crop_start * fs))
            s1 = int(round(crop_end * fs))
            eeg_cropped = data["eeg"][s0:s1]

            mask = data["valid_window_mask"].astype(bool)
            t_win_cropped = data["t_windows"][mask] - crop_start

            states_cropped = data["state_labels"][mask] if "state_labels" in file_keys else None
            conds_cropped = (
                data["condition_labels"][mask] if "condition_labels" in file_keys else None
            )

            sh_cropped = None
            bs_cropped = None
            if "shallow_entry_times" in file_keys:
                sh = data["shallow_entry_times"]
                sh_cropped = sh[(sh >= crop_start) & (sh <= crop_end)] - crop_start
            if "bs_entry_times" in file_keys:
                bs = data["bs_entry_times"]
                bs_cropped = bs[(bs >= crop_start) & (bs <= crop_end)] - crop_start

            hypnotic_type = (
                str(data["hypnotic_type"])
                if "hypnotic_type" in file_keys
                else meta.get("hypnotic_type", "unknown")
            )

            return {
                "sample_id": str(data["sample_id"]),
                "hypnotic_type": hypnotic_type,
                "eeg": eeg_cropped,
                "t_windows": t_win_cropped,
                "state_labels": states_cropped,
                "condition_labels": conds_cropped,
                "shallow_entry_times": sh_cropped,
                "bs_entry_times": bs_cropped,
                "saved_sections": meta.get("saved_sections", {}),
                "metadata": meta,
            }