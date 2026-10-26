with open('digital_twin/app.js', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """      const typeLabel = this.formatAnomalyLabel(atype);"""
replacement1 = """      const isCoasting = (a.fall_tracking_state === 'LOST_COASTING' || a.is_live_observation === false || a.tracking_state === 'LOST_COASTING');
      let typeLabel = this.formatAnomalyLabel(atype);
      if (isCoasting) {
        typeLabel += ' (Coasting)';
      }"""

target2 = """      // Color selection
      let dotColor = isAnomaly ? '#ef4444' : '#38bdf8';
      if (p.anomaly_type === 'FALL_SUSPECTED') dotColor = '#f59e0b';
      if (isSelected) dotColor = '#00f2fe';"""

replacement2 = """      const isCoasting = (p.fall_tracking_state === 'LOST_COASTING' || p.is_live_observation === false);
      // Color selection
      let dotColor = isAnomaly ? '#ef4444' : '#38bdf8';
      if (isCoasting) dotColor = '#f97316';
      else if (p.anomaly_type === 'FALL_SUSPECTED') dotColor = '#f59e0b';
      if (isSelected) dotColor = '#00f2fe';"""

target3 = """      // Pedestrian Center Dot
      ctx.beginPath();
      ctx.arc(mapX, mapY, 5, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Compact Tag Formatting: clean, crisp
      let tagText = `#${tid}`;
      if (isSelected) {
        tagText = `#${tid} [INSPECT]`;
      } else if (isAnomaly) {"""

replacement3 = """      // Pedestrian Center Dot
      ctx.beginPath();
      ctx.arc(mapX, mapY, 5, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.fill();
      if (isCoasting) {
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = '#f97316';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      // Compact Tag Formatting: clean, crisp
      let tagText = `#${tid}`;
      if (isSelected) {
        tagText = `#${tid} [INSPECT]`;
      } else if (isCoasting) {
        tagText = `#${tid} · COASTING`;
      } else if (isAnomaly) {"""

assert target1 in content, "target1 not found"
assert target2 in content, "target2 not found"
assert target3 in content, "target3 not found"

content = content.replace(target1, replacement1, 1)
content = content.replace(target2, replacement2, 1)
content = content.replace(target3, replacement3, 1)

with open('digital_twin/app.js', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("Successfully patched digital_twin/app.js")
