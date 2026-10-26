#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

app_js_path = Path("digital_twin/app.js")
content = app_js_path.read_text(encoding="utf-8")

# 1. Add currentVideoName in constructor
target_const = "this.currentPedestrians = [];"
repl_const = "this.currentPedestrians = [];\n    this.currentVideoName = null;"
if target_const in content:
    content = content.replace(target_const, repl_const, 1)
    print("1. Added currentVideoName in constructor")
else:
    print("Warning: target_const not found")

# 2. Add projectToFloorPlan and clearTrackHistory
target_method = "  resizeCanvases() {"
methods_to_add = """  projectToFloorPlan(rawX, rawY, videoW, videoH) {
    const vw = Math.max(1, videoW || this.videoRefW || 640);
    const vh = Math.max(1, videoH || this.videoRefH || 480);
    const nx = Math.max(0, Math.min(1, rawX / vw));
    const ny = Math.max(0, Math.min(1, rawY / vh));

    const padX = 28;
    const padY = 24;
    const cw = this.canvas.width - padX * 2;
    const ch = this.canvas.height - padY * 2;

    const mapX = padX + nx * cw;
    const mapY = padY + ny * ch;
    return { mapX, mapY, normX: nx, normY: ny, padX, padY, cw, ch };
  }

  clearTrackHistory() {
    this.currentPedestrians = [];
    this.selectedTrackId = null;
    if (this.statDetected) this.statDetected.textContent = '0';
    if (this.statActiveTracks) this.statActiveTracks.textContent = '0';
    if (this.statAnomalies) {
      this.statAnomalies.textContent = '0';
      this.statAnomalies.classList.remove('active');
    }
    if (this.kpiAnomaliesCard) this.kpiAnomaliesCard.classList.remove('active');
    if (this.activeAlertCount) {
      this.activeAlertCount.textContent = '0 ACTIVE';
      this.activeAlertCount.classList.remove('active');
    }
    if (this.alertsList) {
      this.alertsList.innerHTML = `
        <tr class="alert-empty-row">
          <td colspan="8">
            <div class="empty-state">
              <span class="empty-icon">✓</span>
              <span class="empty-text">ALL CLEAR — No abnormal evacuation behaviors detected.</span>
            </div>
          </td>
        </tr>`;
    }
    this.drawDigitalTwinMap([], 1, null, this.selectedDirection);
  }

  resizeCanvases() {"""
if target_method in content:
    content = content.replace(target_method, methods_to_add, 1)
    print("2. Added projectToFloorPlan and clearTrackHistory")
else:
    print("Warning: target_method not found")

# 3. Call clearTrackHistory in stopAnalysisAndChangeVideo
target_stop = "this.selectedTrackId = null;"
# In stopAnalysisAndChangeVideo
if "this.isAnalysisActive = false;\n    this.selectedTrackId = null;" in content:
    content = content.replace(
        "this.isAnalysisActive = false;\n    this.selectedTrackId = null;",
        "this.isAnalysisActive = false;\n    this.clearTrackHistory();",
        1
    )
    print("3. Called clearTrackHistory in stopAnalysisAndChangeVideo")
else:
    print("Warning: stopAnalysisAndChangeVideo pattern not found")

# 4. Handle video_switched in handleWebSocketMessage
target_ws_init = "const data = JSON.parse(event.data);"
repl_ws_init = """const data = JSON.parse(event.data);
          if (data.type === 'video_switched' || (data.video_name && this.currentVideoName && data.video_name !== this.currentVideoName)) {
            this.clearTrackHistory();
          }
          if (data.video_name) {
            this.currentVideoName = data.video_name;
          }"""
if target_ws_init in content:
    content = content.replace(target_ws_init, repl_ws_init, 1)
    print("4. Handled video_switched in handleWebSocketMessage")
else:
    print("Warning: target_ws_init not found")

# 5. Trajectory rendering in drawDigitalTwinMap
old_traj_block = """    // 7. Trajectories first (below pedestrian marker dots)
    const videoRefW = this.videoRefW || 640;
    const videoRefH = this.videoRefH || 480;

    for (const p of pedestrians) {
      const isAnomaly = !!p.anomaly;
      const isSelected = (this.selectedTrackId !== null && p.track_id === this.selectedTrackId);
      const trail = p.trajectory;

      if (trail && trail.length > 1) {
        const recentTrail = trail.slice(-18);
        ctx.beginPath();
        recentTrail.forEach((pt, idx) => {
          const nx = (pt[0] !== undefined ? pt[0] : 0) / videoRefW;
          const ny = (pt[1] !== undefined ? pt[1] : 0) / videoRefH;
          const tx = padX + Math.max(0, Math.min(1, nx)) * cw;
          const ty = padY + Math.max(0, Math.min(1, ny)) * ch;
          if (idx === 0) ctx.moveTo(tx, ty);
          else ctx.lineTo(tx, ty);
        });

        if (isSelected) {
          ctx.strokeStyle = '#00f2fe';
          ctx.lineWidth = 2.0;
        } else if (isAnomaly) {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
          ctx.lineWidth = 1.5;
        } else {
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
          ctx.lineWidth = 1.0;
        }
        ctx.stroke();
      }
    }"""

new_traj_block = """    // 7. Trajectories first (below pedestrian marker dots)
    const videoRefW = this.videoRefW || 640;
    const videoRefH = this.videoRefH || 480;
    const maxDisplacementPx = 65; // px in video space
    const maxFrameGap = 5;        // frame gap threshold to break polyline

    for (const p of pedestrians) {
      const isAnomaly = !!p.anomaly;
      const isSelected = (this.selectedTrackId !== null && p.track_id === this.selectedTrackId);
      const trail = p.trajectory;

      if (trail && trail.length > 1) {
        // Draw only the most recent 10-20 valid observations (max 18 points)
        const recentTrail = trail.slice(-18);

        ctx.beginPath();
        let subpathActive = false;
        let lastPt = null;

        for (let idx = 0; idx < recentTrail.length; idx++) {
          const pt = recentTrail[idx];
          const rawX = pt.x !== undefined ? pt.x : pt[0];
          const rawY = pt.y !== undefined ? pt.y : pt[1];
          const ptFrame = pt.frame !== undefined ? pt.frame : pt[2];

          if (rawX === undefined || rawY === undefined || isNaN(rawX) || isNaN(rawY)) {
            subpathActive = false;
            lastPt = null;
            continue;
          }

          const proj = this.projectToFloorPlan(rawX, rawY, videoRefW, videoRefH);
          const tx = proj.mapX;
          const ty = proj.mapY;

          // Check for displacement jump or frame gap from previous point in trail
          if (lastPt !== null) {
            const dist = Math.hypot(rawX - lastPt.rawX, rawY - lastPt.rawY);
            const frameDiff = (ptFrame !== undefined && lastPt.ptFrame !== undefined) ? (ptFrame - lastPt.ptFrame) : 1;

            if (dist > maxDisplacementPx || frameDiff > maxFrameGap) {
              // Break trail across jump or gap - never connect points across reassignments
              ctx.moveTo(tx, ty);
              lastPt = { rawX, rawY, ptFrame, tx, ty };
              continue;
            }
          }

          if (!subpathActive) {
            ctx.moveTo(tx, ty);
            subpathActive = true;
          } else {
            ctx.lineTo(tx, ty);
          }
          lastPt = { rawX, rawY, ptFrame, tx, ty };
        }

        // Separate polyline per track with thin, low-opacity trails; highlight only inspected track
        if (isSelected) {
          ctx.strokeStyle = '#00f2fe';
          ctx.lineWidth = 2.0;
          ctx.shadowColor = '#00f2fe';
          ctx.shadowBlur = 6;
          ctx.stroke();
          ctx.shadowBlur = 0;
        } else if (isAnomaly) {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.35)';
          ctx.lineWidth = 1.0;
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.20)';
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }
    }"""

if old_traj_block in content:
    content = content.replace(old_traj_block, new_traj_block, 1)
    print("5. Replaced trajectory rendering loop with segmented, jump-protected logic")
else:
    # Try normalizing newlines
    old_norm = old_traj_block.replace("\r\n", "\n")
    content_norm = content.replace("\r\n", "\n")
    if old_norm in content_norm:
        content = content_norm.replace(old_norm, new_traj_block.replace("\r\n", "\n"), 1)
        print("5. Replaced trajectory rendering loop (normalized newlines)")
    else:
        print("Warning: old_traj_block not found")

# 6. Use projectToFloorPlan for pedestrian foot points
old_foot_block = """      const normX = Math.max(0, Math.min(1, rawX / videoRefW));
      const normY = Math.max(0, Math.min(1, rawY / videoRefH));
      const mapX = padX + normX * cw;
      const mapY = padY + normY * ch;"""

new_foot_block = """      const proj = this.projectToFloorPlan(rawX, rawY, videoRefW, videoRefH);
      const mapX = proj.mapX;
      const mapY = proj.mapY;"""

if old_foot_block in content:
    content = content.replace(old_foot_block, new_foot_block, 1)
    print("6. Updated pedestrian foot points to use projectToFloorPlan")
else:
    old_fn = old_foot_block.replace("\r\n", "\n")
    content_n = content.replace("\r\n", "\n")
    if old_fn in content_n:
        content = content_n.replace(old_fn, new_foot_block.replace("\r\n", "\n"), 1)
        print("6. Updated pedestrian foot points (normalized newlines)")
    else:
        print("Warning: old_foot_block not found")

# 7. Update footnote
old_footnote = "ctx.fillText('Camera-to-Plan Normalized Mapping', w - 34, h - 28);"
new_footnote = "ctx.fillText('Camera-to-Plan Normalized Mapping (Approximate / Uncalibrated)', w - 34, h - 28);"
if old_footnote in content:
    content = content.replace(old_footnote, new_footnote, 1)
    print("7. Updated footnote label")
else:
    print("Warning: old_footnote not found")

app_js_path.write_text(content, encoding="utf-8")
print("Saved patched app.js successfully!")
