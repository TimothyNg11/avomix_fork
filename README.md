# AvoMix

Gesture-controlled DJ deck. Mix two tracks with your hands — pinch in mid-air to drag virtual jog wheels, faders, stem mutes, and play/sync buttons rendered over a webcam feed. Built with MediaPipe hand tracking, OpenCV, and a small PyTorch gesture model.

## Demo

Pinch your thumb and index finger to "grab" any on-screen control. Two hands are tracked independently, so you can scratch the left jog wheel while sliding the right volume fader. A peace-sign gesture pauses both decks.

## Features

- **Two decks** with independent play/pause, volume, jog-wheel scrubbing, and per-stem mute (bass, drums, other, vocals).
- **Sync** — match BPM and align drum phase between decks at the tap of a button.
- **Cue points** — set a cue per deck and arm a sync trigger so a deck auto-starts when the other reaches its cue.
- **Echo effect** with a circular delay buffer.
- **Gesture overlay** — a small MLP trained on hand landmarks recognises a peace sign to stop both decks (extendable to more gestures).

## Requirements

- Python 3.10+
- A webcam
- Speakers / headphones

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the runtime app. To retrain the gesture model you'll also need `torch` and `scikit-learn` (already included).

## Songs

Audio is loaded from `songs/<song name>/<stem>.mp3`, where `<stem>` is one of `bass`, `drums`, `other`, `vocals`. The repo doesn't ship audio — drop your own stem-separated tracks in:

```
songs/
  my song/
    bass.mp3
    drums.mp3
    other.mp3
    vocals.mp3
```

To get stems from a regular mp3, run it through a source-separation tool such as [Demucs](https://github.com/facebookresearch/demucs).

The default left/right decks in `main.py` are `clarity` and `stay the night`; change `def_left` / `def_right` in `main.py` to point at folders you actually have. BPM values for sync are hardcoded in `SongSelector.get_bpm` — add your songs there.

## Run

```bash
python main.py
```

The app opens fullscreen. Press `q` to quit.

## Gesture model (optional)

The `avomix_model/` folder contains scripts for training a custom gesture classifier on hand landmarks.

1. `python avomix_model/collect_gestures.py` — record labelled landmark samples (press `R` to record, `N` for a new gesture, `Q` to save).
2. `python avomix_model/train_model.py` — train an MLP on the collected CSV and save `gesture_model.pth` + `gesture_labels.json`.
3. `python avomix_model/recognize_gestures.py` — live-test the trained model.

`main.py` automatically loads `avomix_model/gesture_model.pth` at startup if present, and runs predictions per frame. If the model file is missing it just skips gesture detection — the rest of the app still works.

Note: `collect_gestures.py` and `recognize_gestures.py` load `./hand_landmarker.task` relative to the current working directory, so run them from a directory that contains that file (either repo root or `avomix_model/`).

## Project layout

```
main.py                   # entry point, lays out UI and runs the frame loop
hand_tracking.py          # MediaPipe hand landmarker wrapper + pinch/skeleton drawing
song_selector.py          # audio engine: stem playback, seek, sync, echo, BPM match
ui.py                     # Button, StemButton, VolumeSlider, JogWheel, SyncButton, etc.
hand_landmarker.task      # MediaPipe hand-landmark model
avomix_model/             # gesture-recognition pipeline (collect / train / recognize)
songs/                    # your stem-separated audio (gitignored)
```

## Controls

| Control | Gesture |
|---|---|
| Any button / slider / jog wheel | Pinch thumb + index finger inside the control |
| Drag (slider / jog) | Hold the pinch and move |
| Stop both decks | Show a peace sign to the camera |
| Quit | Press `q` |
