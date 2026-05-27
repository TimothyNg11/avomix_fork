import sounddevice as sd
import soundfile as sf
import numpy as np

STEMS = ["bass", "drums", "other", "vocals"]

class SongSelector:
    def __init__(self, sr=44100):
        self.sr = sr
        self.stems = {"left": [], "right": []}
        self.playing = {"left": False, "right": False}
        self.position = {"left": 0.0, "right": 0.0}
        self.current_song = {"left": None, "right": None}
        self.speed = {"left": 1.0, "right": 1.0}
        self.muted = {"left": [False] * len(STEMS), "right": [False] * len(STEMS)}
        self.volume = {"left": 1.0, "right": 1.0}
        self._seek_delta = {"left": 0.0, "right": 0.0}

        # Cue points (sample position) and sync triggers
        self.cue_point = {"left": None, "right": None}
        # List of triggers: {source_side, source_pos, target_side, target_pos}
        self.sync_triggers = []

        # Echo effect: circular delay buffer per side
        echo_delay_sec = 0.35
        echo_samples = int(sr * echo_delay_sec)
        self.echo_on = {"left": False, "right": False}
        self.echo_decay = 0.45
        self._echo_buf = {
            "left": np.zeros((echo_samples, 2), dtype=np.float32),
            "right": np.zeros((echo_samples, 2), dtype=np.float32),
        }
        self._echo_pos = {"left": 0, "right": 0}

        self.stream = sd.OutputStream(
            samplerate=sr,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, outdata, frames, time, status):
        outdata[:] = 0
        for side in ["left", "right"]:
            if not self.stems[side]:
                continue
            # Apply any pending seek delta first
            seek = self._seek_delta[side]
            if seek != 0.0:
                self._seek_delta[side] = 0.0
                self.position[side] = max(0.0, self.position[side] + seek)
            if not self.playing[side]:
                pass  # skip playback but still run echo below
            else:
                pos = self.position[side]
                max_len = max(len(s) for s in self.stems[side])
                if pos >= max_len:
                    self.playing[side] = False
                elif np.any(pos + np.arange(frames, dtype=np.float64) * self.speed[side] < max_len):
                    speed = self.speed[side]
                    read_positions = pos + np.arange(frames, dtype=np.float64) * speed
                    valid = read_positions < max_len
                    rp = read_positions[valid]
                    i0 = np.floor(rp).astype(np.int64)
                    t = (rp - i0).astype(np.float32)
                    n_valid = len(rp)
                    vol = self.volume[side]
                    for stem_idx, stem_data in enumerate(self.stems[side]):
                        if self.muted[side][stem_idx]:
                            continue
                        slen = len(stem_data)
                        mask = i0 < slen - 1
                        idx = np.minimum(i0[mask], slen - 2)
                        frac = t[mask]
                        samples = stem_data[idx] * (1 - frac[:, None]) + stem_data[idx + 1] * frac[:, None]
                        outdata[:n_valid][mask] += samples * vol
                    self.position[side] = pos + frames * speed
                else:
                    self.playing[side] = False

            # Check sync triggers for this side
            fired = []
            for i, trigger in enumerate(self.sync_triggers):
                if trigger["source_side"] != side or not self.playing[side]:
                    continue
                src_pos = trigger["source_pos"]
                old_pos = self.position[side] - frames
                new_pos = self.position[side]
                if old_pos < src_pos <= new_pos:
                    tgt = trigger["target_side"]
                    self.position[tgt] = float(trigger["target_pos"])
                    self.playing[tgt] = True
                    fired.append(i)
            for i in reversed(fired):
                self.sync_triggers.pop(i)

            # Apply echo effect (runs independently of playback)
            if self.echo_on[side]:
                buf = self._echo_buf[side]
                buf_len = len(buf)
                ep = self._echo_pos[side]
                for i in range(frames):
                    idx = (ep + i) % buf_len
                    outdata[i] += buf[idx] * self.echo_decay
                    buf[idx] = outdata[i]
                self._echo_pos[side] = (ep + frames) % buf_len

        outdata *= 0.5
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def play(self, side):
        self.playing[side] = True

    def pause(self, side):
        self.playing[side] = False

    def toggle_stem(self, side, stem_index):
        self.muted[side][stem_index] = not self.muted[side][stem_index]

    def set_volume(self, side, vol):
        self.volume[side] = max(0.0, min(1.0, vol))

    def seek(self, side, delta_samples):
        if not self.stems[side]:
            return
        self._seek_delta[side] += delta_samples

    def set_cue(self, side):
        """Mark current position as a cue point for this deck."""
        self.cue_point[side] = self.position[side]

    def add_sync_trigger(self, source_side, target_side):
        """Add a trigger: when source reaches its cue, start target at its cue."""
        src_cue = self.cue_point[source_side]
        tgt_cue = self.cue_point[target_side]
        if src_cue is None or tgt_cue is None:
            return False
        self.sync_triggers.append({
            "source_side": source_side,
            "source_pos": src_cue,
            "target_side": target_side,
            "target_pos": tgt_cue,
        })
        return True

    def clear_sync_triggers(self):
        self.sync_triggers.clear()

    def get_bpm(self, song_name):
        BPM_MAP = {
            "Beauty And A Beat": 128.0,
            "clarity": 128.0,
            "i gotta feeling": 128.0,
            "money longer": 135.0,
            "sexyback": 117.0,
            "stay the night": 128.0,
            "want to want me": 114.0,
        }
        return BPM_MAP.get(song_name, 128.0)

    def match_bpm(self, target_side, source_side):
        """Adjusts the speed of target_side to match the BPM of source_side."""
        tgt_song = self.current_song[target_side]
        src_song = self.current_song[source_side]
        if not tgt_song or not src_song:
            return
        
        tgt_bpm = self.get_bpm(tgt_song)
        src_bpm = self.get_bpm(src_song)
        
        self.speed[target_side] = src_bpm / tgt_bpm

    def sync_drums(self, target_side, source_side):
        """Aligns the beat frame (phase) of target_side to source_side."""
        tgt_song = self.current_song.get(target_side)
        src_song = self.current_song.get(source_side)
        if not tgt_song or not src_song:
            return
        
        tgt_bpm = self.get_bpm(tgt_song)
        src_bpm = self.get_bpm(src_song)
        
        tgt_interval = (60.0 / tgt_bpm) * self.sr
        src_interval = (60.0 / src_bpm) * self.sr
        
        src_pos = self.position[source_side]
        tgt_pos = self.position[target_side]
        
        src_phase = (src_pos % src_interval) / src_interval
        tgt_phase = (tgt_pos % tgt_interval) / tgt_interval
        
        diff = src_phase - tgt_phase
        if diff < -0.5:
            diff += 1.0
        elif diff > 0.5:
            diff -= 1.0
            
        delta = diff * tgt_interval
        self.seek(target_side, delta)

    def select(self, side, song):
        self.playing[side] = False
        self.current_song[side] = song
        self.speed[side] = 1.0  # Reset speed on load
        loaded = []
        for stem in STEMS:
            data, _ = sf.read(f"songs/{song}/{stem}.mp3", dtype="float32")
            if data.ndim == 1:
                data = np.column_stack([data, data])
            loaded.append(data)
        self.stems[side] = loaded
        self.position[side] = 0.0
        self.muted[side] = [False] * len(STEMS)

    def close(self):
        self.stream.stop()
        self.stream.close()
