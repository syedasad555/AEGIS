with open('scripts/fall_detection.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """        if len(hist) >= 3:
            oldest_cy = hist[0][2]
            cy_drop_amount = (cy - oldest_cy) / float(frame_h)
            if cy_drop_amount > self.centroid_drop_threshold:
                dynamic_drop = True
                cur_state["has_dynamic_drop"] = True"""

replacement1 = """        if len(hist) >= 3:
            recent_idx = max(0, len(hist) - 5)
            recent_cy = hist[recent_idx][2]
            cy_drop_amount = (cy - recent_cy) / float(frame_h)
            if cy_drop_amount > self.centroid_drop_threshold:
                dynamic_drop = True
                cur_state["has_dynamic_drop"] = True"""

target2 = """            # Decrement fall evidence gracefully rather than immediately resetting
            cur_state["consecutive_fall_frames"] = max(0, cur_state["consecutive_fall_frames"] - 1)"""

replacement2 = """            # Decrement fall evidence gracefully rather than immediately resetting
            cur_state["consecutive_fall_frames"] = max(0, cur_state["consecutive_fall_frames"] - 1)
            if cur_state["consecutive_upright_frames"] >= 8:
                cur_state["has_dynamic_drop"] = False"""

assert target1 in content, "target1 not found"
assert target2 in content, "target2 not found"

content = content.replace(target1, replacement1, 1)
content = content.replace(target2, replacement2, 1)

with open('scripts/fall_detection.py', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("Successfully updated scripts/fall_detection.py")
