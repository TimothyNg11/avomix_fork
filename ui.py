import cv2
import math


class Button:
    def __init__(self, x, y, width, height, color=(0, 0, 0), active_color=(0, 255, 0)):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.color = color
        self.active_color = active_color
        self.pinched = {"Left": False, "Right": False}
        self.on = False

    def contains(self, pos):
        if pos is None:
            return False
        px, py = pos
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def update(self, hand, pos):
        inside = self.contains(pos)
        if inside and not self.pinched[hand]:
            self.pinched[hand] = True
            self.on = not self.on
            if self.on:
                self.activate()
            else:
                self.deactivate()
            return True
        if not inside:
            self.pinched[hand] = False
        return False

    def draw(self, frame):
        color = self.active_color if self.on else self.color
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            3,
        )

    def activate(self):
        pass

    def deactivate(self):
        pass


class PlayButton(Button):
    """Tap to play/pause left or right song."""

    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, **kwargs)
        self.selector = selector
        self.side = side

    def activate(self):
        self.selector.play(self.side)

    def deactivate(self):
        self.selector.pause(self.side)

    def draw(self, frame):
        super().draw(frame)
        label = "PLAY" if not self.on else "PAUSE"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        ty = self.y + (self.height + text_size[1]) // 2
        text_color = (0, 0, 0) # Black text everywhere
        cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, thickness)


class StemButton(Button):
    """Toggle a specific stem on/off for a deck."""

    STEM_COLORS = {
        "bass": (255, 100, 50),
        "drums": (50, 150, 255),
        "other": (50, 255, 150),
        "vocals": (200, 100, 255),
    }

    def __init__(self, x, y, width, height, selector, side, stem_index, label, **kwargs):
        color = StemButton.STEM_COLORS.get(label, (0, 200, 200))
        super().__init__(x, y, width, height, active_color=color, **kwargs)
        self.selector = selector
        self.side = side
        self.stem_index = stem_index
        self.label = label
        self.on = True  # stems start enabled (not muted)

    def activate(self):
        self.selector.muted[self.side][self.stem_index] = False

    def deactivate(self):
        self.selector.muted[self.side][self.stem_index] = True

    def draw(self, frame):
        color = self.active_color if self.on else (60, 60, 60)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            -1 if self.on else 2,
        )
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (255, 255, 255),
            1,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(self.label, font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        ty = self.y + (self.height + text_size[1]) // 2
        text_color = (0, 0, 0)
        cv2.putText(frame, self.label, (tx, ty), font, font_scale, text_color, thickness)


class EchoButton(Button):
    """Toggle echo effect for a deck."""

    def __init__(self, x, y, width, height, selector, side, **kwargs):
        super().__init__(x, y, width, height, active_color=(0, 180, 255), **kwargs)
        self.selector = selector
        self.side = side

    def activate(self):
        self.selector.echo_on[self.side] = True

    def deactivate(self):
        self.selector.echo_on[self.side] = False
        # Clear the buffer so old echo doesn't linger
        self.selector._echo_buf[self.side][:] = 0

    def draw(self, frame):
        color = self.active_color if self.on else (60, 60, 60)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            -1 if self.on else 2,
        )
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (255, 255, 255),
            1,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = "ECHO"
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        ty = self.y + (self.height + text_size[1]) // 2
        text_color = (0, 0, 0)
        cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, thickness)


class VolumeSlider:
    """Vertical volume slider controlled by pinch drag."""

    def __init__(self, x, y, width, height, selector, side):
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.selector = selector
        self.side = side
        self.volume = 1.0
        self.grabbed = {"Left": False, "Right": False}

    def contains(self, pos):
        if pos is None:
            return False
        px, py = pos
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def update(self, hand, pos):
        if not self.grabbed[hand]:
            if self.contains(pos):
                self.grabbed[hand] = True

        if self.grabbed[hand] and pos is not None:
            _, py = pos
            self.volume = 1.0 - (py - self.y) / self.height
            self.volume = max(0.0, min(1.0, self.volume))
            self.selector.set_volume(self.side, self.volume)

    def release(self, hand):
        self.grabbed[hand] = False

    def draw(self, frame):
        # Background track
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (60, 60, 60),
            -1,
        )
        # Filled portion (bottom-up)
        fill_h = int(self.height * self.volume)
        fill_y = self.y + self.height - fill_h
        cv2.rectangle(
            frame,
            (self.x, fill_y),
            (self.x + self.width, self.y + self.height),
            (0, 220, 0),
            -1,
        )
        # Border
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (255, 255, 255),
            2,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize("VOL", font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        cv2.putText(frame, "VOL", (tx, self.y - 8), font, font_scale, (0, 0, 0), thickness)
        # Percentage
        pct = f"{int(self.volume * 100)}%"
        pct_font_scale = 0.6
        pct_thickness = 2
        text_size_pct = cv2.getTextSize(pct, font, pct_font_scale, pct_thickness)[0]
        tx_pct = self.x + (self.width - text_size_pct[0]) // 2
        cv2.putText(frame, pct, (tx_pct, self.y + self.height + 25), font, pct_font_scale, (0, 0, 0), pct_thickness)


class JogWheel:
    """Circular jog wheel for scrubbing through a track."""

    def __init__(self, cx, cy, radius, selector, side):
        self.cx, self.cy = cx, cy
        self.radius = radius
        self.selector = selector
        self.side = side
        self.angle = 0.0
        self.grabbed = {"Left": False, "Right": False}
        self.last_angle = {"Left": None, "Right": None}
        self.samples_per_radian = selector.sr * 2  # ~2 seconds per radian
        self.cue_angle = None  # angle at which cue was set

    def contains(self, pos):
        if pos is None:
            return False
        px, py = pos
        dist = math.sqrt((px - self.cx) ** 2 + (py - self.cy) ** 2)
        return dist <= self.radius

    def update(self, hand, pos):
        if not self.grabbed[hand]:
            if self.contains(pos):
                self.grabbed[hand] = True
                if pos:
                    self.last_angle[hand] = math.atan2(pos[1] - self.cy, pos[0] - self.cx)
            return

        if pos is None:
            return

        current_angle = math.atan2(pos[1] - self.cy, pos[0] - self.cx)
        if self.last_angle[hand] is not None:
            delta = current_angle - self.last_angle[hand]
            # Normalize to [-pi, pi]
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi

            self.angle += delta
            self.selector.seek(self.side, delta * self.samples_per_radian)

        self.last_angle[hand] = current_angle

    def release(self, hand):
        self.grabbed[hand] = False
        self.last_angle[hand] = None

    def draw(self, frame):
        grabbed_any = any(self.grabbed.values())
        ring_color = (0, 200, 200) if grabbed_any else (180, 180, 180)
        cv2.circle(frame, (self.cx, self.cy), self.radius, ring_color, 2)
        # Inner fill
        cv2.circle(frame, (self.cx, self.cy), self.radius - 8, (40, 40, 40), -1)
        cv2.circle(frame, (self.cx, self.cy), self.radius - 8, (100, 100, 100), 1)
        # Rotating indicator line
        lx = int(self.cx + (self.radius - 12) * math.cos(self.angle))
        ly = int(self.cy + (self.radius - 12) * math.sin(self.angle))
        cv2.line(frame, (self.cx, self.cy), (lx, ly), (255, 255, 255), 2)
        # Center dot
        cv2.circle(frame, (self.cx, self.cy), 4, (255, 255, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = self.side.upper()
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = self.cx - text_size[0] // 2
        ty = self.cy + self.radius + 28
        cv2.putText(frame, label, (tx, ty), font, font_scale, (0, 0, 0), thickness)


class CueButton(Button):
    """Set a cue point at the current playback position for a deck."""

    def __init__(self, x, y, width, height, selector, side, jog_wheel=None, **kwargs):
        super().__init__(x, y, width, height, active_color=(0, 200, 255), **kwargs)
        self.selector = selector
        self.side = side
        self.jog_wheel = jog_wheel

    def activate(self):
        self.selector.set_cue(self.side)
        if self.jog_wheel:
            self.jog_wheel.cue_angle = self.jog_wheel.angle

    def deactivate(self):
        # Tapping again updates the cue point
        self.selector.set_cue(self.side)
        if self.jog_wheel:
            self.jog_wheel.cue_angle = self.jog_wheel.angle
        self.on = True  # keep visually "on" once a cue is set

    def draw(self, frame):
        has_cue = self.selector.cue_point[self.side] is not None
        color = self.active_color if has_cue else (60, 60, 60)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            -1 if has_cue else 2,
        )
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (255, 255, 255),
            1,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = "CUE"
        font_scale = 0.6
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        ty = self.y + (self.height + text_size[1]) // 2
        text_color = (0, 0, 0) if has_cue else (180, 180, 180)
        cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, thickness)
        # Show cue time
        if has_cue:
            cue_pos = self.selector.cue_point[self.side]
            secs = cue_pos / self.selector.sr
            mins = int(secs // 60)
            s = int(secs % 60)
            hundredths = int((secs - int(secs)) * 100)
            time_label = f"{mins}:{s:02d}.{hundredths:02d}"
            time_scale = 0.4
            time_thickness = 1
            ts = cv2.getTextSize(time_label, font, time_scale, time_thickness)[0]
            cv2.putText(frame, time_label, (self.x + (self.width - ts[0]) // 2, self.y + self.height + 15), font, time_scale, (200, 200, 200), time_thickness)


class SyncButton(Button):
    """Arm a sync trigger: when left deck hits its cue, right deck auto-starts at its cue (and vice versa)."""

    def __init__(self, x, y, width, height, selector, **kwargs):
        super().__init__(x, y, width, height, active_color=(255, 50, 100), **kwargs)
        self.selector = selector

    def activate(self):
        # Create triggers in both directions
        self.selector.clear_sync_triggers()
        ok1 = self.selector.add_sync_trigger("left", "right")
        ok2 = self.selector.add_sync_trigger("right", "left")
        if not ok1 and not ok2:
            self.on = False  # no cue points set, can't arm

    def deactivate(self):
        self.selector.clear_sync_triggers()

    def draw(self, frame):
        armed = self.on and len(self.selector.sync_triggers) > 0
        if self.on and not armed:
            self.on = False  # triggers were consumed, auto-disarm
        color = self.active_color if self.on else (60, 60, 60)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            -1 if armed else 2,
        )
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (255, 255, 255),
            1,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = "SYNC"
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        ty = self.y + (self.height + text_size[1]) // 2
        text_color = (0, 0, 0)
        cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, thickness)

class FullSyncButton(Button):
    """Match this deck's BPM and align drum beats to the other deck."""
    def __init__(self, x, y, width, height, selector, side, other_side, **kwargs):
        super().__init__(x, y, width, height, active_color=(255, 150, 50), **kwargs)
        self.selector = selector
        self.side = side
        self.other_side = other_side

    def activate(self):
        self.selector.match_bpm(self.side, self.other_side)
        self.selector.sync_drums(self.side, self.other_side)

    def deactivate(self):
        # Reset to native speed
        self.selector.speed[self.side] = 1.0

    def draw(self, frame):
        color = self.active_color if self.on else (60, 60, 60)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            color,
            -1 if self.on else 2,
        )
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            (255, 255, 255),
            1,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = "SYNC"
        font_scale = 0.7
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = self.x + (self.width - text_size[0]) // 2
        ty = self.y + (self.height + text_size[1]) // 2
        text_color = (0, 0, 0)
        cv2.putText(frame, label, (tx, ty), font, font_scale, text_color, thickness)
