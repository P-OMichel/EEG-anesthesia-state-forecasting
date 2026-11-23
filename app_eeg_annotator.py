import sys
from PyQt6.QtWidgets import QApplication

from Annotation_App.processing import EEGProcessor
from Annotation_App.annotations import AnnotationStore
from Annotation_App.ui_window import EEGAnnotatorWindow


def main():
    fs = 128
    win_sec = 30.0
    step_sec = 10.0

    app = QApplication(sys.argv)

    processor = EEGProcessor(fs=fs)
    store = AnnotationStore(fs=fs, win_sec=win_sec, step_sec=step_sec)
    window = EEGAnnotatorWindow(processor=processor, store=store)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()