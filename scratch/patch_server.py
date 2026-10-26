with open('server.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """                # Clean, compact label
                if is_anom:"""

replacement = """                # Clean, compact label
                if is_coasting:
                    lbl = f"#{tid} - FALLEN (COASTING)"
                elif is_anom:"""

if target in content:
    content = content.replace(target, replacement, 1)
    with open('server.py', 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    print("Successfully patched server.py")
else:
    print("Target not found in server.py")
