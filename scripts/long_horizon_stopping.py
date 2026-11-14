import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

class LongHorizonStoppingVerifier:
    """
    Maintains long-horizon spatial trajectory history for each track and computes
    net displacement over the stopping candidate duration (e.g. 3.0s / 75 frames).
    
    If net displacement over the candidate stopping duration >= cumulative_disp_thresh_px
    (default 20.0px, or normalized relative to frame height), the track is physically
    progressing (e.g. slowly walking in a crowd bottleneck) and is NOT stationary.
    """
    def __init__(self, threshold_sec: float = 3.0, max_net_disp_px: float = 20.0):
        self.threshold_sec = threshold_sec
        self.max_net_disp_px = max_net_disp_px
        # track_id -> deque of (timestamp, x, y)
        self.history: Dict[int, deque] = {}

    def reset(self):
        self.history.clear()

    def update(self, track_id: int, pos: Tuple[float, float], timestamp: float):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=150)
        self.history[track_id].append((timestamp, pos[0], pos[1]))

    def verify_stopping(self, track_id: int, current_timestamp: float, current_pos: Tuple[float, float], scale_factor: float = 1.0) -> Tuple[bool, float]:
        """
        Returns (is_genuinely_stationary, net_displacement_over_window).
        """
        hist = self.history.get(track_id)
        if not hist or len(hist) < 2:
            return True, 0.0
            
        target_time = current_timestamp - self.threshold_sec
        # Find oldest point at or after target_time, or the earliest point if window is shorter
        ref_pt = hist[0]
        for item in hist:
            if item[0] >= target_time:
                ref_pt = item
                break
                
        dx = current_pos[0] - ref_pt[1]
        dy = current_pos[1] - ref_pt[2]
        net_disp = math.hypot(dx, dy)
        
        effective_max_disp = self.max_net_disp_px * scale_factor
        is_stationary = (net_disp < effective_max_disp)
        return is_stationary, net_disp

