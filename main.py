import cv2
from hand_tracking import HandTracker, draw_hand_skeleton
from song_selector import SongSelector, STEMS
import os
import sys
from ui import PlayButton, StemButton, VolumeSlider, JogWheel, FullSyncButton

# Add avomix_model to path to allow importing recognizing_gestures
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'avomix_model'))
from recognize_gestures import load_model, normalize_landmarks
import torch

def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)

    # Load the gesture recognition model
    try:
        gesture_model, gesture_labels, device = load_model()
    except Exception as e:
        print(f"Warning: Could not load gesture model: {e}")
        gesture_model = None

    if not cap.isOpened():

        print("Error: Camera not found.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from camera.")
        return

    height, width, _ = frame.shape

    def_left = "clarity"
    def_right = "stay the night"

    song_selector = SongSelector()
    song_selector.select("left", def_left)
    song_selector.select("right", def_right)

    # --- Layout ---
    left_cx = width // 4
    right_cx = 3 * width // 4

    # Row 1: Jog wheels (top area)
    jog_radius = int(width * 0.08)
    jog_cy = int(height * 0.22)
    left_jog = JogWheel(left_cx, jog_cy, jog_radius, song_selector, "left")
    right_jog = JogWheel(right_cx, jog_cy, jog_radius, song_selector, "right")
    jog_wheels = [left_jog, right_jog]

    # Row 2: Stem toggle buttons (well below jog wheels)
    stem_btn_w, stem_btn_h = int(width * 0.1), int(height * 0.08)
    stem_gap = int(width * 0.015)
    stem_row_w = len(STEMS) * stem_btn_w + (len(STEMS) - 1) * stem_gap
    stem_y = jog_cy + jog_radius + int(height * 0.08)

    left_stem_x = left_cx - stem_row_w // 2
    right_stem_x = right_cx - stem_row_w // 2

    stem_buttons = []
    for i, stem_name in enumerate(STEMS):
        lx = left_stem_x + i * (stem_btn_w + stem_gap)
        stem_buttons.append(StemButton(lx, stem_y, stem_btn_w, stem_btn_h, song_selector, "left", i, stem_name))
        rx = right_stem_x + i * (stem_btn_w + stem_gap)
        stem_buttons.append(StemButton(rx, stem_y, stem_btn_w, stem_btn_h, song_selector, "right", i, stem_name))

    # Row 3: Play buttons + Volume sliders + FullSync buttons (bottom)
    play_w, play_h = int(width * 0.12), int(height * 0.15)
    slider_w, slider_h = int(width * 0.05), int(height * 0.25)
    full_sync_w, full_sync_h = int(width * 0.12), int(height * 0.1)

    play_y = stem_y + stem_btn_h + int(height * 0.14)
    slider_y = play_y - int(height * 0.04)

    # Left Deck
    left_slider_x = left_cx - int(width * 0.22)
    left_full_sync_x = left_cx + int(width * 0.08)
    # Center play button exactly between slider and full sync
    left_play_x = left_slider_x + slider_w + (left_full_sync_x - (left_slider_x + slider_w)) // 2 - (play_w // 2)
    left_full_sync_y = play_y + int((play_h - full_sync_h) // 2)

    left_slider = VolumeSlider(left_slider_x, slider_y, slider_w, slider_h, song_selector, "left")
    left_btn = PlayButton(left_play_x, play_y, play_w, play_h, song_selector, "left")
    left_full_sync = FullSyncButton(left_full_sync_x, left_full_sync_y, full_sync_w, full_sync_h, song_selector, "left", "right")

    # Right Deck
    right_full_sync_x = right_cx - full_sync_w - int(width * 0.08)
    right_full_sync_y = play_y + int((play_h - full_sync_h) // 2)
    
    right_slider_x = right_cx + int(width * 0.22) - slider_w
    # Center play button exactly between full sync and slider
    right_play_x = right_full_sync_x + full_sync_w + (right_slider_x - (right_full_sync_x + full_sync_w)) // 2 - (play_w // 2)

    right_slider = VolumeSlider(right_slider_x, slider_y, slider_w, slider_h, song_selector, "right")
    right_btn = PlayButton(right_play_x, play_y, play_w, play_h, song_selector, "right")
    right_full_sync = FullSyncButton(right_full_sync_x, right_full_sync_y, full_sync_w, full_sync_h, song_selector, "right", "left")

    sliders = [left_slider, right_slider]

    buttons = [left_btn, right_btn, left_full_sync, right_full_sync] + stem_buttons

    # Create a fullscreen window
    cv2.namedWindow('CV DJ Set', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('CV DJ Set', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("DJ Hand Tracking Started. Press 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.detect_async(rgb_frame)

            result = tracker.get_latest_result()

            frame = draw_hand_skeleton(frame, tracker, result)

            if gesture_model is not None and result and result.hand_landmarks:
                for lms in result.hand_landmarks:
                    features = normalize_landmarks(lms, width, height)
                    tensor = torch.from_numpy(features.astype("float32")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = gesture_model(tensor)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred_idx = probs.max(1)
                        if conf.item() > 0.99:
                            predicted = gesture_labels[pred_idx.item()]
                            if predicted == "peace_sign":
                                left_btn.on = False
                                right_btn.on = False
                                left_btn.deactivate()
                                right_btn.deactivate()

            for hand in ["Left", "Right"]:
                pinch_pos = tracker.pinch_pos[hand]
                # Flip to display coords for hit detection
                if pinch_pos:
                    pinch_pos = (width - 1 - pinch_pos[0], pinch_pos[1])

                for button in buttons:
                    if tracker.state[hand] == 1:
                        button.update(hand, pinch_pos)
                    else:
                        button.pinched[hand] = False

                for slider in sliders:
                    if tracker.state[hand] == 1:
                        slider.update(hand, pinch_pos)
                    else:
                        slider.release(hand)

                for jog in jog_wheels:
                    if tracker.state[hand] == 1:
                        jog.update(hand, pinch_pos)
                    else:
                        jog.release(hand)

            reversed_frame = cv2.flip(frame, 1)

            for button in buttons:
                button.draw(reversed_frame)
            for slider in sliders:
                slider.draw(reversed_frame)
            for jog in jog_wheels:
                jog.draw(reversed_frame)

            # Draw current playback time for each deck
            font = cv2.FONT_HERSHEY_SIMPLEX
            for side, cx in [("left", left_cx), ("right", right_cx)]:
                if song_selector.stems[side]:
                    pos_sec = song_selector.position[side] / song_selector.sr
                    total_sec = max(len(s) for s in song_selector.stems[side]) / song_selector.sr
                    cur_m, cur_s = int(pos_sec // 60), pos_sec % 60
                    cur_hundredths = int((cur_s - int(cur_s)) * 100)
                    cur_s_int = int(cur_s)
                    tot_m, tot_s = int(total_sec // 60), total_sec % 60
                    tot_hundredths = int((tot_s - int(tot_s)) * 100)
                    tot_s_int = int(tot_s)
                    time_str = f"{cur_m}:{cur_s_int:02d}.{cur_hundredths:02d} / {tot_m}:{tot_s_int:02d}.{tot_hundredths:02d}"
                    font_scale = 0.7
                    thickness = 2
                    ts = cv2.getTextSize(time_str, font, font_scale, thickness)[0]
                    tx = cx - ts[0] // 2
                    ty = height - 25
                    cv2.putText(reversed_frame, time_str, (tx, ty), font, font_scale, (220, 220, 220), thickness)

            cv2.imshow('CV DJ Set', reversed_frame)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        song_selector.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
