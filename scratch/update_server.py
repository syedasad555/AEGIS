with open('server.py', 'rb') as f:
    raw = f.read()

# Let's inspect raw around anom_map
idx = raw.find(b'anom_map = {')
print("Found anom_map at byte offset:", idx)

# Let's replace the lines using line-based replacement
lines = raw.splitlines(keepends=True)

# Find line with b'# If selected, draw corner accent brackets'
start_line = None
for i, line in enumerate(lines):
    if b'# If selected, draw corner accent brackets' in line:
        start_line = i
        break

print("start_line:", start_line)

# Let's construct the replacement block as UTF-8 bytes
replacement_lines = [
    b'                # If selected or fall detected, draw corner accent brackets\r\n',
    b'                if is_selected or anom_type == "FALL_DETECTED":\r\n',
    b'                    corner_col = (0, 240, 255) if is_selected else (50, 50, 235)\r\n',
    b'                    c_len = min(12, bw // 3, bh // 3)\r\n',
    b'                    cv2.line(annotated_frame, (x1 - 3, y1 - 3), (x1 - 3 + c_len, y1 - 3), corner_col, 2)\r\n',
    b'                    cv2.line(annotated_frame, (x1 - 3, y1 - 3), (x1 - 3, y1 - 3 + c_len), corner_col, 2)\r\n',
    b'                    cv2.line(annotated_frame, (x2 + 3, y2 + 3), (x2 + 3 - c_len, y2 + 3), corner_col, 2)\r\n',
    b'                    cv2.line(annotated_frame, (x2 + 3, y2 + 3), (x2 + 3 - c_len, y2 + 3), corner_col, 2)\r\n',
    b'\r\n',
    b'                # Draw bottom foot position marker\r\n',
    b'                foot_x, foot_y = (x1 + x2) // 2, y2\r\n',
    b'                cv2.circle(annotated_frame, (foot_x, foot_y), 3, box_color, -1, cv2.LINE_AA)\r\n',
    b'\r\n',
    b'                # Clean, compact label\r\n',
    b'                if is_anom:\r\n',
    b'                    anom_map = {\r\n',
    b'                        "FALL_DETECTED": "FALL DETECTED",\r\n',
    b'                        "FALL_SUSPECTED": "FALL SUSPECTED",\r\n',
    b'                        "PROLONGED_STOPPING": "STOPPING",\r\n',
    b'                        "WRONG_DIRECTION": "WRONG DIR",\r\n',
    b'                        "RESTRICTED_ZONE_ENTRY": "RESTRICTED"\r\n',
    b'                    }\r\n',
    b'                    status_str = anom_map.get(ped.get("anomaly_type", ""), "ALERT")\r\n',
    b'                    lbl = f"#{tid} - {status_str}"\r\n',
    b'                else:\r\n',
    b'                    pdir = ped.get("direction", "STATIONARY")\r\n',
    b'                    lbl = f"#{tid} - {pdir}" if pdir != "STATIONARY" else f"#{tid}"\r\n',
]

# In original lines, start_line to start_line+24
for j in range(start_line, start_line + 25):
    print(f"{j}: {lines[j]}")

# Replace lines[start_line : start_line + 24] with replacement_lines
new_lines = lines[:start_line] + replacement_lines + lines[start_line + 24:]
with open('server.py', 'wb') as f:
    f.writelines(new_lines)
print("Updated server.py successfully!")
