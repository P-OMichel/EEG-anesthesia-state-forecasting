# EEG Depth & Event Annotator (128 Hz)

A desktop **PyQt6 + PyQtGraph** application for visualizing 128 Hz EEG recordings, inspecting spectrograms and sliding-window power, cropping valid signal intervals, and manually annotating depth states, clinical conditions, and entry events for supervised learning.

---

## Project Structure

```text
project_root/
├── main.py                         # Application entry point
└── Annotation_App/
    ├── processing.py               # EEG extraction, NaN cleaning, spectrogram, power & initial state estimation
    ├── annotations.py              # Label state store, crop logic, auto-resume detection & .npz I/O
    └── ui_window.py                # PyQt6 GUI, synchronized plots & interactive drag/box-select tools