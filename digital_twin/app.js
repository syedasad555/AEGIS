/**
 * AEGIS Evacuation Vision Console — Front-end Application Logic
 * ===========================================================================
 * Features:
 *   - Video-Selection-First Workflow with Preloaded Presets & Custom File Upload
 *   - Real-Time Live AI Stream (WebSocket) with YOLOv8n + ByteTrack Sensor Feed
 *   - Architectural 2D Digital Twin Virtual Map (Recognizable Corridor Layout)
 *   - Anti-Collision Pedestrian Label & Leader-Line Resolution
 *   - Explainable Evacuation Anomaly Rules (WRONG_DIRECTION, PROLONGED_STOPPING, RESTRICTED_ZONE_ENTRY)
 *   - Interactive Alert Click-to-Inspect Synchronization across CCTV & Digital Twin
 *   - Defensible Severity Classification (CRITICAL / WARNING) & Duration Tracking
 *   - Responsive 100vh Desktop Layout (1366×768 / 1920×1080)
 */

class DigitalTwinApp {
  constructor() {
    // Initial Video Selection Screen elements
    this.videoSelectionScreen = document.getElementById('videoSelectionScreen');
    this.presetCards = document.querySelectorAll('.preset-card');
    this.videoFileInput = document.getElementById('videoFileInput');
    this.selectedFileName = document.getElementById('selectedFileName');
    this.customPathInput = document.getElementById('customPathInput');
    this.validatePathBtn = document.getElementById('validatePathBtn');
    this.initialDirectionSelect = document.getElementById('initialDirectionSelect');
    this.selectionValidationMsg = document.getElementById('selectionValidationMsg');
    this.startAnalysisBtn = document.getElementById('startAnalysisBtn');
    this.activeVideoName = document.getElementById('activeVideoName');
    this.changeVideoBtn = document.getElementById('changeVideoBtn');

    // Left panel elements (CCTV feed)
    this.cctvContainer = document.getElementById('cctvContainer');
    this.cctvLiveStream = document.getElementById('cctvLiveStream');
    this.videoEl = document.getElementById('cctvVideo');
    this.cctvOverlay = document.getElementById('cctvOverlay');
    this.overlayCtx = this.cctvOverlay ? this.cctvOverlay.getContext('2d') : null;
    this.leftPanelBadge = document.getElementById('leftPanelBadge');
    this.cctvErrorBanner = document.getElementById('cctvErrorBanner');
    this.cctvErrorDetail = document.getElementById('cctvErrorDetail');
    this.retryVideoBtn = document.getElementById('retryVideoBtn');

    // Right panel (Architectural 2D Digital Twin)
    this.canvas = document.getElementById('digitalTwinCanvas');
    this.ctx = this.canvas.getContext('2d');

    // Header controls
    this.directionSelect = document.getElementById('directionSelect');
    this.togglePauseBtn = document.getElementById('togglePauseBtn');
    this.pauseIcon = document.getElementById('pauseIcon');
    this.pauseText = document.getElementById('pauseText');
    this.statusText = document.getElementById('statusText');
    this.connectionStatus = document.getElementById('connectionStatus');

    // Metric stat elements (People Detected, Active Alerts, Processing FPS)
    this.statDetected = document.getElementById('statDetected');
    this.statAnomalies = document.getElementById('statAnomalies');
    this.statFps = document.getElementById('statFps');
    this.kpiAnomaliesCard = document.getElementById('kpiAnomaliesCard');

    // Alert console elements
    this.alertsList = document.getElementById('alertsList');
    this.activeAlertCount = document.getElementById('activeAlertCount');
    this.clearSelectionBtn = document.getElementById('clearSelectionBtn');

    // Modal & Toast elements
    this.zonesModal = document.getElementById('zonesModal');
    this.openZonesModalBtn = document.getElementById('openZonesModalBtn');
    this.closeZonesModalBtn = document.getElementById('closeZonesModalBtn');
    this.saveZonesBtn = document.getElementById('saveZonesBtn');
    this.resetZonesBtn = document.getElementById('resetZonesBtn');
    this.cfgSpeedThreshold = document.getElementById('cfgSpeedThreshold');
    this.cfgStoppingFrames = document.getElementById('cfgStoppingFrames');
    this.cfgZonesJson = document.getElementById('cfgZonesJson');
    this.toastContainer = document.getElementById('toastContainer');

    // Architectural Zones Definition (Normalized 0.0 - 1.0)
    this.zonesConfig = {
      restricted: [
        { name: 'RESTRICTED SECURITY ZONE', points: [[0.30, 0.22], [0.68, 0.22], [0.68, 0.54], [0.30, 0.54]] }
      ],
      evacuation_route: [
        { name: 'MAIN EVACUATION CORRIDOR', points: [[0.06, 0.16], [0.94, 0.16], [0.94, 0.84], [0.06, 0.84]] }
      ],
      safe_zone: [
        { name: 'SAFE ASSEMBLY REFUGE', points: [[0.74, 0.06], [0.96, 0.06], [0.96, 0.32], [0.74, 0.32]] }
      ],
      exit: [
        { name: 'EMERGENCY FIRE EXIT', points: [[0.90, 0.42], [0.97, 0.42], [0.97, 0.64], [0.90, 0.64]] }
      ]
    };

    // Stream and rendering dimensions
    this.videoRefW = 640;
    this.videoRefH = 480;
    this.isPaused = false;
    this.ws = null;
    this.pulseAngle = 0;
    this.flowAnimOffset = 0;
    this.currentPedestrians = [];
    this.currentVideoName = null;

    // Application state
    this.selectedVideoPath = 'assets/evacuation_test_corridor.avi';
    this.selectedVideoName = 'evacuation_test_corridor.avi';
    this.selectedDirection = 'LEFT';
    this.isAnalysisActive = false;

    this.clearTrackHistory();
    this.init();
  }

  async init() {
    this.resizeCanvases();
    window.addEventListener('resize', () => this.resizeCanvases());

    // Setup Video Selection Listeners
    this.setupSelectionListeners();

    // Direction selector change in topbar
    if (this.directionSelect) {
      this.directionSelect.value = this.selectedDirection;
      this.directionSelect.addEventListener('change', (e) => {
        this.selectedDirection = e.target.value;
        this.switchDirection(this.selectedDirection);
      });
    }

    // Pause / Resume button
    if (this.togglePauseBtn) {
      this.togglePauseBtn.addEventListener('click', () => {
        this.togglePause();
      });
    }

    // Stop / Change Video button
    if (this.changeVideoBtn) {
      this.changeVideoBtn.addEventListener('click', () => {
        this.stopAnalysisAndChangeVideo();
      });
    }

    if (this.retryVideoBtn) {
      this.retryVideoBtn.addEventListener('click', () => {
        this.stopAnalysisAndChangeVideo();
      });
    }

    // Clear track inspection selection
    if (this.clearSelectionBtn) {
      this.clearSelectionBtn.addEventListener('click', () => {
        this.selectTrack(null);
      });
    }

    // Zones & Rules Modal Listeners
    if (this.openZonesModalBtn) {
      this.openZonesModalBtn.addEventListener('click', () => this.openZonesModal());
    }
    if (this.closeZonesModalBtn) {
      this.closeZonesModalBtn.addEventListener('click', () => this.closeZonesModal());
    }
    if (this.saveZonesBtn) {
      this.saveZonesBtn.addEventListener('click', () => this.saveZonesConfig());
    }
    if (this.resetZonesBtn) {
      this.resetZonesBtn.addEventListener('click', () => this.resetZonesToDefault());
    }

    // Close modal on outside click
    if (this.zonesModal) {
      this.zonesModal.addEventListener('click', (e) => {
        if (e.target === this.zonesModal) this.closeZonesModal();
      });
    }

    // Continuous rendering animation loop for smooth UI pulses and flow arrows
    this.startAnimationLoop();

    // Connect to WebSocket server
    this.connectWebSocket();
  }

  setupSelectionListeners() {
    // Preset Scenario Selection Cards
    this.presetCards.forEach((card) => {
      card.addEventListener('click', () => {
        this.presetCards.forEach((c) => c.classList.remove('active'));
        card.classList.add('active');

        this.selectedVideoPath = card.getAttribute('data-video');
        this.selectedVideoName = this.selectedVideoPath.split('/').pop();
        const preferredDir = card.getAttribute('data-dir') || 'LEFT';
        this.selectedDirection = preferredDir;

        if (this.initialDirectionSelect) {
          this.initialDirectionSelect.value = preferredDir;
        }
        if (this.directionSelect) {
          this.directionSelect.value = preferredDir;
        }
        if (this.selectedFileName) {
          this.selectedFileName.textContent = `Preset selected: ${this.selectedVideoName}`;
        }
        this.hideValidationMsg();
      });
    });

    // Custom File Input
    if (this.videoFileInput) {
      this.videoFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        this.selectedFileName.textContent = `Uploading ${file.name}...`;
        this.showValidationMsg(`Uploading ${file.name} to server...`, 'info');

        try {
          const res = await fetch(`/api/upload_video?filename=${encodeURIComponent(file.name)}`, {
            method: 'POST',
            body: file
          });
          const data = await res.json();

          if (data.status === 'ok' && data.valid) {
            this.selectedVideoPath = data.path;
            this.selectedVideoName = data.filename;
            this.presetCards.forEach((c) => c.classList.remove('active'));
            if (this.selectedFileName) {
              this.selectedFileName.textContent = `${data.filename} (${data.width}×${data.height}, ${data.frames} frames)`;
            }
            this.showValidationMsg(`✓ Video uploaded and verified: ${data.filename} (${data.width}×${data.height}, ${data.fps} FPS)`, 'success');
          } else {
            this.showValidationMsg(`✕ Upload failed: ${data.error || 'Invalid video format'}`, 'error');
            this.selectedFileName.textContent = 'Upload failed. Select a preset or another file.';
          }
        } catch (err) {
          this.showValidationMsg(`✕ Upload error: ${err.message}`, 'error');
        } finally {
          this.videoFileInput.value = '';
        }
      });
    }

    // Custom Server Path Input
    if (this.validatePathBtn && this.customPathInput) {
      const doValidate = async () => {
        const path = this.customPathInput.value.trim();
        if (!path) {
          this.showValidationMsg('Please enter a server path to test.', 'error');
          return;
        }

        this.showValidationMsg(`Verifying ${path}...`, 'info');
        try {
          const res = await fetch('/api/validate_video', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_path: path })
          });
          const data = await res.json();

          if (data.valid) {
            this.selectedVideoPath = data.path;
            this.selectedVideoName = data.filename;
            this.presetCards.forEach((c) => c.classList.remove('active'));
            if (this.selectedFileName) {
              this.selectedFileName.textContent = `${data.filename} (${data.width}×${data.height}, ${data.total_frames} frames)`;
            }
            this.showValidationMsg(`✓ Video verified: ${data.filename} (${data.width}×${data.height}, ${data.fps} FPS, ${data.total_frames} frames)`, 'success');
          } else {
            this.showValidationMsg(`✕ Cannot open video: ${data.error}`, 'error');
          }
        } catch (err) {
          this.showValidationMsg(`✕ Validation request failed: ${err.message}`, 'error');
        }
      };

      this.validatePathBtn.addEventListener('click', doValidate);
      this.customPathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          doValidate();
        }
      });
    }

    // Evacuation direction dropdown in initial screen
    if (this.initialDirectionSelect) {
      this.initialDirectionSelect.addEventListener('change', (e) => {
        this.selectedDirection = e.target.value;
        if (this.directionSelect) {
          this.directionSelect.value = this.selectedDirection;
        }
      });
    }

    // Start Analysis button
    if (this.startAnalysisBtn) {
      this.startAnalysisBtn.addEventListener('click', async () => {
        await this.startAnalysis();
      });
    }
  }

  showValidationMsg(text, type = 'info') {
    if (!this.selectionValidationMsg) return;
    this.selectionValidationMsg.textContent = text;
    this.selectionValidationMsg.className = `validation-msg ${type}`;
    this.selectionValidationMsg.style.display = 'block';
  }

  hideValidationMsg() {
    if (!this.selectionValidationMsg) return;
    this.selectionValidationMsg.style.display = 'none';
  }

  async startAnalysis() {
    this.showValidationMsg(`Initializing analysis for ${this.selectedVideoName}...`, 'info');

    // Pre-flight check via /api/validate_video
    try {
      const res = await fetch('/api/validate_video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_path: this.selectedVideoPath })
      });
      const data = await res.json();

      if (!data.valid) {
        this.showValidationMsg(`✕ Cannot start analysis: ${data.error}`, 'error');
        return;
      }

      this.videoRefW = data.width || 640;
      this.videoRefH = data.height || 480;
    } catch (e) {
      console.warn('Pre-flight validation note:', e);
    }

    // Hide selection screen smoothly
    if (this.videoSelectionScreen) {
      this.videoSelectionScreen.classList.add('hidden');
    }

    this.isAnalysisActive = true;
    this.selectedTrackId = null;

    // Update Topbar UI
    if (this.activeVideoName) {
      this.activeVideoName.textContent = this.selectedVideoName;
    }
    if (this.directionSelect) {
      this.directionSelect.value = this.selectedDirection;
    }
    if (this.cctvErrorBanner) {
      this.cctvErrorBanner.style.display = 'none';
    }

    // Start Video Stream over WebSocket
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        action: 'start_stream',
        video_path: this.selectedVideoPath,
        direction: this.selectedDirection
      }));
    } else {
      this.connectWebSocket();
    }

    this.showToast(`Active monitoring started: ${this.selectedVideoName}`, 'info');
  }

  stopAnalysisAndChangeVideo() {
    this.isAnalysisActive = false;
    this.selectedTrackId = null;

    // Send stop command to server
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'stop_stream' }));
    }

    if (this.videoEl) {
      this.videoEl.pause();
    }

    // Reset metrics
    if (this.statDetected) this.statDetected.textContent = '0';
    if (this.statAnomalies) {
      this.statAnomalies.textContent = '0';
      this.statAnomalies.classList.remove('active');
    }
    if (this.statFps) this.statFps.textContent = '-- FPS';
    if (this.kpiAnomaliesCard) this.kpiAnomaliesCard.classList.remove('active');
    if (this.activeAlertCount) {
      this.activeAlertCount.textContent = '0';
      this.activeAlertCount.classList.remove('active');
    }

    if (this.alertsList) {
      this.alertsList.innerHTML = `
        <div class="incident-empty">
          <span class="empty-icon">✓</span>
          <span class="empty-text">Select a video to begin</span>
        </div>`;
    }

    // Show selection screen
    if (this.videoSelectionScreen) {
      this.videoSelectionScreen.classList.remove('hidden');
    }

    this.hideValidationMsg();
    this.resizeCanvases();
  }

  projectToFloorPlan(rawX, rawY, videoW, videoH) {
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
    if (this.statAnomalies) {
      this.statAnomalies.textContent = '0';
      this.statAnomalies.classList.remove('active');
    }
    if (this.kpiAnomaliesCard) this.kpiAnomaliesCard.classList.remove('active');
    if (this.activeAlertCount) {
      this.activeAlertCount.textContent = '0';
      this.activeAlertCount.classList.remove('active');
    }
    if (this.alertsList) {
      this.alertsList.innerHTML = `
        <div class="incident-empty">
          <span class="empty-icon">✓</span>
          <span class="empty-text">No active alerts</span>
        </div>`;
    }
    if (this.ctx && this.canvas && this.zonesConfig && this.zonesConfig.evacuation_route) {
      this.drawDigitalTwinMap([], 1, null, this.selectedDirection);
    }
  }

  resizeCanvases() {
    // Resize Digital Twin Virtual Map Canvas
    const parentDT = this.canvas.parentElement;
    if (parentDT) {
      const rect = parentDT.getBoundingClientRect();
      this.canvas.width = rect.width || 640;
      this.canvas.height = rect.height || 480;
    }

    // Resize CCTV Overlay Canvas to match video dimensions
    if (this.cctvOverlay && this.cctvOverlay.parentElement) {
      const parentCCTV = this.cctvOverlay.parentElement;
      const rect = parentCCTV.getBoundingClientRect();
      this.cctvOverlay.width = rect.width || 640;
      this.cctvOverlay.height = rect.height || 480;
    }

    // Redraw map with current state
    this.drawDigitalTwinMap(this.currentPedestrians, 1, null, this.selectedDirection);
  }

  connectWebSocket() {
    try {
      this.ws = new WebSocket('ws://localhost:8765');

      this.ws.onopen = () => {
        if (this.statusText) this.statusText.textContent = 'WS LIVE';
        if (this.connectionStatus) {
          this.connectionStatus.style.borderColor = 'rgba(16, 185, 129, 0.4)';
          this.connectionStatus.style.color = '#10b981';
        }

        // If user already initialized analysis, send stream command
        if (this.isAnalysisActive) {
          this.ws.send(JSON.stringify({
            action: 'start_stream',
            video_path: this.selectedVideoPath,
            direction: this.selectedDirection
          }));
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'video_switched' || (data.video_name && this.currentVideoName && data.video_name !== this.currentVideoName)) {
            this.clearTrackHistory();
          }
          if (data.video_name) {
            this.currentVideoName = data.video_name;
          }
          if (data.type === 'init') {
            if (data.expected_direction && this.directionSelect) {
              this.directionSelect.value = data.expected_direction;
              this.selectedDirection = data.expected_direction;
            }
          } else if (data.type === 'frame_update') {
            if (this.statusText) this.statusText.textContent = 'WS STREAM';
            this.renderLiveFrame(data);
          } else if (data.type === 'error') {
            this.showCctvError(data.message);
          } else if (data.type === 'stream_stopped') {
            if (this.statusText) this.statusText.textContent = 'STANDBY';
          }
        } catch (err) {
          console.error('Error parsing WS message:', err);
        }
      };

      this.ws.onclose = () => {
        if (this.statusText) this.statusText.textContent = 'DISCONNECTED';
        if (this.connectionStatus) {
          this.connectionStatus.style.borderColor = 'rgba(239, 68, 68, 0.4)';
          this.connectionStatus.style.color = '#ef4444';
        }
        setTimeout(() => this.connectWebSocket(), 2500);
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket connection note:', err);
      };
    } catch (e) {
      console.warn('WS Init exception:', e);
    }
  }

  showCctvError(message) {
    if (this.cctvErrorBanner && this.cctvErrorDetail) {
      this.cctvErrorDetail.textContent = message;
      this.cctvErrorBanner.style.display = 'flex';
      if (this.cctvLiveStream) this.cctvLiveStream.style.display = 'none';
      if (this.cctvVideo) this.cctvVideo.style.display = 'none';
    }
  }

  switchDirection(direction) {
    this.selectedDirection = direction;
    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isAnalysisActive) {
      this.ws.send(JSON.stringify({
        action: 'switch_video',
        video_path: this.selectedVideoPath,
        direction: direction
      }));
    }
    this.drawDigitalTwinMap(this.currentPedestrians, 1, null, direction);
  }

  togglePause() {
    this.isPaused = !this.isPaused;
    if (this.pauseIcon && this.pauseText) {
      this.pauseIcon.textContent = this.isPaused ? '▶' : '⏸';
      this.pauseText.textContent = this.isPaused ? 'Resume' : 'Pause';
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ action: 'toggle_pause' }));
    }

    if (this.videoEl) {
      if (this.isPaused) this.videoEl.pause();
      else this.videoEl.play().catch(() => {});
    }
  }

  selectTrack(trackId) {
    if (this.selectedTrackId === trackId) {
      this.selectedTrackId = null;
    } else {
      this.selectedTrackId = trackId;
    }

    // Update clear button visibility
    if (this.clearSelectionBtn) {
      this.clearSelectionBtn.style.display = (this.selectedTrackId !== null) ? 'inline-block' : 'none';
    }

    // Send selection to server over WebSocket
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        action: 'select_track',
        track_id: this.selectedTrackId
      }));
    }

    // Update compact incident list selection state
    if (this.alertsList) {
      this.alertsList.querySelectorAll('[data-track-id]').forEach((el) => {
        const tid = parseInt(el.getAttribute('data-track-id'), 10);
        el.classList.toggle('selected', tid === this.selectedTrackId);
      });
    }

    // Redraw Digital Twin canvas immediately
    this.drawDigitalTwinMap(this.currentPedestrians, 1, null, this.selectedDirection);
  }

  startAnimationLoop() {
    const loop = () => {
      this.pulseAngle = (this.pulseAngle + 0.05) % (Math.PI * 2);
      this.flowAnimOffset = (this.flowAnimOffset + 0.5) % 24;
      if (this.currentPedestrians.length > 0) {
        this.drawDigitalTwinMap(this.currentPedestrians, 1, null, this.selectedDirection);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  renderLiveFrame(data) {
    // 1. Display Live Stream Frame on Left Panel
    if (data.frame_image && this.cctvLiveStream) {
      if (this.cctvLiveStream.style.display !== 'block') {
        this.cctvLiveStream.style.display = 'block';
        if (this.videoEl) this.videoEl.style.display = 'none';
        if (this.cctvOverlay) this.cctvOverlay.style.display = 'none';
        if (this.cctvErrorBanner) this.cctvErrorBanner.style.display = 'none';
      }
      this.cctvLiveStream.src = data.frame_image;
    }

    // 2. Store dynamic dimensions from incoming feed
    if (data.width && data.height) {
      this.videoRefW = data.width;
      this.videoRefH = data.height;
    }

    // 3. Update Metrics
    const pedestrians = data.pedestrians || [];
    this.currentPedestrians = pedestrians;
    const anomalies = pedestrians.filter((p) => p.anomaly);

    if (this.statDetected) this.statDetected.textContent = pedestrians.length;
    if (this.statAnomalies) {
      this.statAnomalies.textContent = anomalies.length;
      this.statAnomalies.classList.toggle('active', anomalies.length > 0);
    }
    if (this.kpiAnomaliesCard) {
      this.kpiAnomaliesCard.classList.toggle('active', anomalies.length > 0);
    }
    // Processing FPS = actual pipeline throughput (not source video FPS)
    if (this.statFps) {
      const procFps = (data.processing_fps !== undefined && data.processing_fps !== null)
        ? Number(data.processing_fps)
        : null;
      this.statFps.textContent = (procFps !== null && !Number.isNaN(procFps))
        ? `${procFps.toFixed(1)} FPS`
        : '-- FPS';
    }

    // 4. Update Alerts Console Feed
    this.updateAlertsList(anomalies, data.active_alerts);

    // 5. Render Synchronized 2D Digital Twin Map on Right Panel
    const direction = data.expected_direction || this.selectedDirection;
    this.drawDigitalTwinMap(pedestrians, data.frame || 1, data.zones, direction);
  }

  formatTimestamp(sec) {
    if (sec === undefined || sec === null) return '00:00.0';
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1);
    const mStr = String(m).padStart(2, '0');
    const sStr = String(s).padStart(4, '0');
    return `${mStr}:${sStr}`;
  }

  formatAnomalyLabel(type) {
    const map = {
      FALL_DETECTED: 'Fallen Pedestrian',
      FALL_SUSPECTED: 'Suspected Fall',
      PROLONGED_STOPPING: 'Prolonged Stopping',
      WRONG_DIRECTION: 'Wrong Direction Flow',
      RESTRICTED_ZONE_ENTRY: 'Restricted Zone Intrusion'
    };
    return map[type] || type.replace(/_/g, ' ');
  }

  updateAlertsList(anomalies, activeAlertsData) {
    if (!this.alertsList) return;

    if (anomalies.length === 0) {
      if (this.activeAlertCount) {
        this.activeAlertCount.textContent = '0';
        this.activeAlertCount.classList.remove('active');
      }
      this.alertsList.innerHTML = `
        <div class="incident-empty">
          <span class="empty-icon">✓</span>
          <span class="empty-text">No active alerts</span>
        </div>`;
      return;
    }

    if (this.activeAlertCount) {
      this.activeAlertCount.textContent = String(anomalies.length);
      this.activeAlertCount.classList.add('active');
    }

    const priorityMap = {
      FALL_DETECTED: 10,
      FALL_SUSPECTED: 8,
      RESTRICTED_ZONE_ENTRY: 6,
      PROLONGED_STOPPING: 4,
      WRONG_DIRECTION: 2
    };
    const sortedAnomalies = [...anomalies].sort((a, b) => {
      const pa = priorityMap[a.anomaly_type] || 0;
      const pb = priorityMap[b.anomaly_type] || 0;
      return pb - pa;
    });

    let html = '';
    for (const a of sortedAnomalies) {
      const tid = a.track_id;
      const isSel = (tid === this.selectedTrackId);
      const atype = a.anomaly_type || 'ANOMALOUS_MOVEMENT';
      const severity = a.severity || (atype === 'FALL_DETECTED' || atype === 'RESTRICTED_ZONE_ENTRY' || (a.stopping_duration_frames >= 90) ? 'CRITICAL' : 'WARNING');
      const sevClass = (severity === 'CRITICAL') ? 'critical' : 'warning';
      const duration = a.duration_sec ? `${a.duration_sec}s` : (a.stopping_duration_sec ? `${a.stopping_duration_sec}s` : 'live');
      const ts = this.formatTimestamp(a.timestamp);
      const isCoasting = (a.fall_tracking_state === 'LOST_COASTING' || a.is_live_observation === false || a.tracking_state === 'LOST_COASTING');
      let typeLabel = this.formatAnomalyLabel(atype);
      if (isCoasting) typeLabel += ' · coasting';

      html += `
        <div class="incident-item ${sevClass} ${isSel ? 'selected' : ''}" data-track-id="${tid}">
          <div class="incident-id">#${tid}</div>
          <div class="incident-main">
            <div class="incident-type">${typeLabel}</div>
            <div class="incident-meta">${ts} · ${duration}</div>
          </div>
          <div class="incident-severity ${sevClass}">${severity}</div>
        </div>`;
    }

    this.alertsList.innerHTML = html;

    this.alertsList.querySelectorAll('[data-track-id]').forEach((el) => {
      el.addEventListener('click', () => {
        const tid = parseInt(el.getAttribute('data-track-id'), 10);
        this.selectTrack(tid);
      });
    });
  }

  /* ==========================================================================
     High-Fidelity Architectural 2D Digital Twin Map Canvas Renderer
     ========================================================================== */

  drawDigitalTwinMap(pedestrians, currentFrame = 1, zonesData = null, expectedDirection = 'LEFT') {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const ctx = this.ctx;

    // 1. Clear background (Deep Architectural Slate)
    ctx.fillStyle = '#080c16';
    ctx.fillRect(0, 0, w, h);

    // 2. Architectural Floor Grid
    ctx.strokeStyle = '#101726';
    ctx.lineWidth = 1;
    const gridSize = 32;
    for (let x = 0; x < w; x += gridSize) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // 3. Architectural Building Walls & Corridor Boundaries
    const padX = 28;
    const padY = 24;
    const cw = w - padX * 2;
    const ch = h - padY * 2;

    // Outer double concrete wall structure
    ctx.fillStyle = '#121a2d';
    ctx.fillRect(padX - 8, padY - 8, cw + 16, ch + 16);
    ctx.fillStyle = '#080c16';
    ctx.fillRect(padX, padY, cw, ch);

    ctx.strokeStyle = '#273852';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(padX - 8, padY - 8, cw + 16, ch + 16);
    ctx.strokeRect(padX, padY, cw, ch);

    // Doorway / Ingress & Egress Openings in Corridor Walls
    // West Entrance Opening
    ctx.fillStyle = '#080c16';
    ctx.fillRect(padX - 9, padY + ch * 0.40, 10, ch * 0.25);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(padX - 9, padY + ch * 0.40, 2, ch * 0.25);

    // East Emergency Exit Opening
    ctx.fillStyle = '#080c16';
    ctx.fillRect(padX + cw - 1, padY + ch * 0.35, 10, ch * 0.30);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.strokeRect(padX + cw + 7, padY + ch * 0.35, 2, ch * 0.30);

    // Corridor Central Walkway Guide Lane
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.12)';
    ctx.lineWidth = 1;
    ctx.setLineDash([8, 8]);
    ctx.beginPath();
    ctx.moveTo(padX, padY + ch * 0.5);
    ctx.lineTo(padX + cw, padY + ch * 0.5);
    ctx.stroke();
    ctx.setLineDash([]);

    // 4. Render Zones with Architectural Refinement
    // Main Evacuation Corridor
    this.drawArchitecturalZone(ctx, padX, padY, cw, ch, this.zonesConfig.evacuation_route[0], 'rgba(16, 185, 129, 0.05)', '#10b981', 'MAIN EVACUATION CORRIDOR');
    // Restricted Security Zone
    this.drawArchitecturalZone(ctx, padX, padY, cw, ch, this.zonesConfig.restricted[0], 'rgba(239, 68, 68, 0.09)', '#ef4444', 'RESTRICTED SECURITY ZONE (DANGER)');
    // Safe Assembly Area
    this.drawArchitecturalZone(ctx, padX, padY, cw, ch, this.zonesConfig.safe_zone[0], 'rgba(56, 189, 248, 0.08)', '#38bdf8', 'SAFE REFUGE AREA');
    // Emergency Exit Zone
    this.drawArchitecturalZone(ctx, padX, padY, cw, ch, this.zonesConfig.exit[0], 'rgba(245, 158, 11, 0.14)', '#f59e0b', 'EMERGENCY EXIT PORTAL');

    // 5. Dynamic Evacuation Flow Chevrons along Corridor
    this.drawCorridorFlowChevrons(ctx, padX, padY, cw, ch, expectedDirection);

    // Quiet doorway hints
    ctx.fillStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.font = '600 9px sans-serif';
    ctx.fillText('ENTRY', padX + 6, padY + ch * 0.38);
    ctx.fillStyle = 'rgba(245, 158, 11, 0.65)';
    ctx.fillText('EXIT', padX + cw - 36, padY + ch * 0.33);

    // Compact footnote: approximate projection
    this.drawMapLegendAndFootnote(ctx, w, h);

    if (!pedestrians || pedestrians.length === 0) return;

    // 7. Trajectories first (below pedestrian marker dots)
    const videoRefW = this.videoRefW || 640;
    const videoRefH = this.videoRefH || 480;
    const maxDisplacementPx = 65; // px in video space (aligned with config trajectory_max_displacement_px)
    const maxFrameGap = 10;       // aligned with config trajectory_max_frame_gap

    for (const p of pedestrians) {
      const isAnomaly = !!p.anomaly;
      const isSelected = (this.selectedTrackId !== null && p.track_id === this.selectedTrackId);
      const trail = p.trajectory;

      // Reduce visual clutter: only draw trails for selected or anomalous tracks
      if (!(isSelected || isAnomaly)) continue;

      if (trail && trail.length > 1) {
        // Draw only the most recent observations (max 12 points)
        const recentTrail = trail.slice(-12);

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

        if (isSelected) {
          ctx.strokeStyle = 'rgba(0, 242, 254, 0.85)';
          ctx.lineWidth = 1.6;
          ctx.stroke();
        } else {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.30)';
          ctx.lineWidth = 1.0;
          ctx.stroke();
        }
      }
    }

    // 8. Render Pedestrian Marker Dots & Anti-Collision Tags
    const occupiedLabelRects = [];
    const sortedPedestrians = [...pedestrians].sort((a, b) => {
      const ya = (a.bbox ? a.bbox[1] : (a.y || 0));
      const yb = (b.bbox ? b.bbox[1] : (b.y || 0));
      return ya - yb;
    });

    for (const p of sortedPedestrians) {
      const tid = p.track_id;
      const isAnomaly = !!p.anomaly;
      const isSelected = (this.selectedTrackId !== null && tid === this.selectedTrackId);

      // Map bottom-center foot coordinates onto floor plan
      let rawX, rawY;
      if (p.bbox && p.bbox.length === 4) {
        rawX = (p.bbox[0] + p.bbox[2]) / 2;
        rawY = p.bbox[3];
      } else {
        rawX = p.x !== undefined ? p.x : (p.foot ? p.foot[0] : 0);
        rawY = p.y !== undefined ? p.y : (p.foot ? p.foot[1] : 0);
      }

      const proj = this.projectToFloorPlan(rawX, rawY, videoRefW, videoRefH);
      const mapX = proj.mapX;
      const mapY = proj.mapY;

      const isCoasting = (p.fall_tracking_state === 'LOST_COASTING' || p.is_live_observation === false);
      // Color selection
      let dotColor = isAnomaly ? '#ef4444' : '#38bdf8';
      if (isCoasting) dotColor = '#f97316';
      else if (p.anomaly_type === 'FALL_SUSPECTED') dotColor = '#f59e0b';
      if (isSelected) dotColor = '#00f2fe';

      // Anomaly Pulse Halo
      if (isAnomaly) {
        const pulseRadius = 10 + Math.sin(this.pulseAngle) * 3;
        ctx.beginPath();
        ctx.arc(mapX, mapY, pulseRadius, 0, Math.PI * 2);
        ctx.strokeStyle = (p.anomaly_type === 'FALL_SUSPECTED') ? 'rgba(245, 158, 11, 0.6)' : 'rgba(239, 68, 68, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Selected Track Inspection Brackets & Pulsing Ring
      if (isSelected) {
        const selRadius = 13 + Math.sin(this.pulseAngle * 1.5) * 2;
        ctx.beginPath();
        ctx.arc(mapX, mapY, selRadius, 0, Math.PI * 2);
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // 4 Corner Inspection Brackets
        const bSize = 6;
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 1.5;
        // TL
        ctx.beginPath(); ctx.moveTo(mapX - selRadius, mapY - selRadius + bSize); ctx.lineTo(mapX - selRadius, mapY - selRadius); ctx.lineTo(mapX - selRadius + bSize, mapY - selRadius); ctx.stroke();
        // TR
        ctx.beginPath(); ctx.moveTo(mapX + selRadius - bSize, mapY - selRadius); ctx.lineTo(mapX + selRadius, mapY - selRadius); ctx.lineTo(mapX + selRadius, mapY - selRadius + bSize); ctx.stroke();
        // BL
        ctx.beginPath(); ctx.moveTo(mapX - selRadius, mapY + selRadius - bSize); ctx.lineTo(mapX - selRadius, mapY + selRadius); ctx.lineTo(mapX - selRadius + bSize, mapY + selRadius); ctx.stroke();
        // BR
        ctx.beginPath(); ctx.moveTo(mapX + selRadius - bSize, mapY + selRadius); ctx.lineTo(mapX + selRadius, mapY + selRadius); ctx.lineTo(mapX + selRadius, mapY + selRadius - bSize); ctx.stroke();
      }

      // Pedestrian Center Dot
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
        tagText = `#${tid}`;
      } else if (isCoasting) {
        tagText = `#${tid}`;
      } else if (isAnomaly) {
        const shortMap = {
          FALL_DETECTED: 'FALL',
          FALL_SUSPECTED: 'FALL?',
          PROLONGED_STOPPING: 'STOP',
          WRONG_DIRECTION: 'REV',
          RESTRICTED_ZONE_ENTRY: 'ZONE'
        };
        const sType = shortMap[p.anomaly_type] || 'ALRT';
        tagText = `#${tid} · ${sType}`;
      }

      ctx.font = '600 9.5px sans-serif';
      const tw = ctx.measureText(tagText).width;
      const tagW = tw + 8;
      const tagH = 16;

      // Candidate Label Positions around the dot
      const candidates = [
        { x: mapX + 8, y: mapY - 8, needLeader: false },
        { x: mapX - tagW - 8, y: mapY - 8, needLeader: false },
        { x: mapX + 8, y: mapY - tagH - 8, needLeader: true },
        { x: mapX - tagW - 8, y: mapY - tagH - 8, needLeader: true },
        { x: mapX - tagW / 2, y: mapY + 10, needLeader: true },
        { x: mapX + 8, y: mapY - tagH * 2 - 10, needLeader: true }
      ];

      let chosen = null;
      for (const cand of candidates) {
        const cx1 = Math.max(padX + 2, Math.min(padX + cw - tagW - 2, cand.x));
        const cy1 = Math.max(padY + 2, Math.min(padY + ch - tagH - 2, cand.y));
        const cx2 = cx1 + tagW;
        const cy2 = cy1 + tagH;

        let collides = false;
        for (const r of occupiedLabelRects) {
          if (!(cx2 < r.x1 || cx1 > r.x2 || cy2 < r.y1 || cy1 > r.y2)) {
            collides = true;
            break;
          }
        }

        if (!collides) {
          chosen = { x1: cx1, y1: cy1, x2: cx2, y2: cy2, needLeader: cand.needLeader };
          break;
        }
      }

      if (!chosen) {
        const cx1 = Math.max(padX + 2, Math.min(padX + cw - tagW - 2, mapX + 8));
        const cy1 = Math.max(padY + 2, Math.min(padY + ch - tagH - 2, mapY - 8));
        chosen = { x1: cx1, y1: cy1, x2: cx1 + tagW, y2: cy1 + tagH, needLeader: false };
      }

      occupiedLabelRects.push(chosen);

      // Leader line if tag is offset
      if (chosen.needLeader) {
        ctx.beginPath();
        ctx.moveTo(mapX, mapY);
        ctx.lineTo(chosen.x1 + tagW / 2, chosen.y1 + tagH / 2);
        ctx.strokeStyle = isSelected ? 'rgba(0, 242, 254, 0.6)' : (isAnomaly ? 'rgba(239, 68, 68, 0.45)' : 'rgba(56, 189, 248, 0.35)');
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Tag Badge Background
      ctx.fillStyle = isSelected ? 'rgba(10, 24, 38, 0.95)' : 'rgba(10, 15, 26, 0.9)';
      ctx.fillRect(chosen.x1, chosen.y1, tagW, tagH);
      ctx.strokeStyle = isSelected ? '#00f2fe' : (p.anomaly_type === 'FALL_SUSPECTED' ? '#f59e0b' : (isAnomaly ? '#ef4444' : 'rgba(56, 189, 248, 0.5)'));
      ctx.lineWidth = isSelected ? 1.5 : 1;
      ctx.strokeRect(chosen.x1, chosen.y1, tagW, tagH);

      // Tag Text
      ctx.fillStyle = isSelected ? '#38bdf8' : (p.anomaly_type === 'FALL_SUSPECTED' ? '#fde68a' : (isAnomaly ? '#fca5a5' : '#f8fafc'));
      ctx.fillText(tagText, chosen.x1 + 4, chosen.y1 + 11.5);
    }
  }

  drawArchitecturalZone(ctx, padX, padY, cw, ch, zone, fillStyle, strokeStyle, label) {
    if (!zone || !zone.points) return;

    ctx.beginPath();
    zone.points.forEach((pt, idx) => {
      const px = padX + pt[0] * cw;
      const py = padY + pt[1] * ch;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();

    ctx.fillStyle = fillStyle;
    ctx.fill();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Small architectural corner ticks
    zone.points.forEach((pt) => {
      const px = padX + pt[0] * cw;
      const py = padY + pt[1] * ch;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px - 2, py - 2, 4, 4);
    });

    // Label
    if (label && zone.points.length > 0) {
      const lx = padX + zone.points[0][0] * cw + 6;
      const ly = padY + zone.points[0][1] * ch + 13;
      ctx.fillStyle = strokeStyle;
      ctx.font = '700 9px sans-serif';
      ctx.fillText(label, lx, ly);
    }
  }

  drawCorridorFlowChevrons(ctx, padX, padY, cw, ch, direction) {
    const centerY = padY + ch * 0.5;
    ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;

    // Chevrons across the corridor length
    const chevronCount = 8;
    const spacing = cw / (chevronCount + 1);

    for (let i = 1; i <= chevronCount; i++) {
      let cx = padX + i * spacing;
      let cy = centerY;

      if (direction === 'LEFT') {
        cx -= this.flowAnimOffset;
        if (cx < padX + 20) cx += cw - 40;
        // Draw left arrow <
        ctx.beginPath();
        ctx.moveTo(cx + 6, cy - 5);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + 6, cy + 5);
        ctx.stroke();
      } else if (direction === 'RIGHT') {
        cx += this.flowAnimOffset;
        if (cx > padX + cw - 20) cx -= cw - 40;
        // Draw right arrow >
        ctx.beginPath();
        ctx.moveTo(cx - 6, cy - 5);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx - 6, cy + 5);
        ctx.stroke();
      } else if (direction === 'UP') {
        cy -= this.flowAnimOffset;
        if (cy < padY + 20) cy += ch - 40;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy + 6);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + 5, cy + 6);
        ctx.stroke();
      } else if (direction === 'DOWN') {
        cy += this.flowAnimOffset;
        if (cy > padY + ch - 20) cy -= ch - 40;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy - 6);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + 5, cy - 6);
        ctx.stroke();
      }
    }
  }

  drawMapLegendAndFootnote(ctx, w, h) {
    const lx = 28;
    const ly = h - 22;
    ctx.fillStyle = 'rgba(10, 15, 26, 0.75)';
    ctx.fillRect(lx - 4, ly - 8, 168, 16);

    ctx.font = '500 9px sans-serif';
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath(); ctx.arc(lx + 4, ly, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Person', lx + 11, ly + 3);

    ctx.fillStyle = '#ef4444';
    ctx.beginPath(); ctx.arc(lx + 58, ly, 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Alert', lx + 65, ly + 3);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.55)';
    ctx.font = '500 8.5px monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Approximate projection (uncalibrated)', w - 28, h - 14);
    ctx.textAlign = 'left';
  }

  /* ==========================================================================
     Modal & Config Management
     ========================================================================== */

  openZonesModal() {
    if (this.zonesModal) {
      this.zonesModal.classList.add('active');
      if (this.cfgZonesJson) {
        this.cfgZonesJson.value = JSON.stringify(this.zonesConfig, null, 2);
      }
    }
  }

  closeZonesModal() {
    if (this.zonesModal) {
      this.zonesModal.classList.remove('active');
    }
  }

  async saveZonesConfig() {
    try {
      const newZones = JSON.parse(this.cfgZonesJson.value);
      this.zonesConfig = newZones;

      const payload = {
        zones: newZones,
        direction: this.selectedDirection,
        speed_threshold: parseFloat(this.cfgSpeedThreshold.value) || 0.6,
        stopping_frames: parseInt(this.cfgStoppingFrames.value, 10) || 30
      };

      const res = await fetch('/api/save_zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        this.showToast('Evacuation zones & detection thresholds updated successfully', 'success');
        this.closeZonesModal();
      } else {
        this.showToast('Failed to save zones to server', 'error');
      }
    } catch (e) {
      this.showToast('Invalid JSON format: ' + e.message, 'error');
    }
  }

  resetZonesToDefault() {
    this.zonesConfig = {
      restricted: [
        { name: 'RESTRICTED SECURITY ZONE', points: [[0.30, 0.22], [0.68, 0.22], [0.68, 0.54], [0.30, 0.54]] }
      ],
      evacuation_route: [
        { name: 'MAIN EVACUATION CORRIDOR', points: [[0.06, 0.16], [0.94, 0.16], [0.94, 0.84], [0.06, 0.84]] }
      ],
      safe_zone: [
        { name: 'SAFE REFUGE AREA', points: [[0.74, 0.06], [0.96, 0.06], [0.96, 0.32], [0.74, 0.32]] }
      ],
      exit: [
        { name: 'EMERGENCY FIRE EXIT', points: [[0.90, 0.42], [0.97, 0.42], [0.97, 0.64], [0.90, 0.64]] }
      ]
    };
    if (this.cfgZonesJson) {
      this.cfgZonesJson.value = JSON.stringify(this.zonesConfig, null, 2);
    }
    if (this.cfgSpeedThreshold) this.cfgSpeedThreshold.value = '0.6';
    if (this.cfgStoppingFrames) this.cfgStoppingFrames.value = '30';
    this.showToast('Reset zones configuration to factory defaults', 'info');
  }

  showToast(message, type = 'info') {
    if (!this.toastContainer) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    this.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(4px)';
      toast.style.transition = 'all 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, 3200);
  }
}

// Instantiate application on DOM ready or immediately if already loaded
function initDigitalTwinApp() {
  if (!window.app) {
    window.app = new DigitalTwinApp();
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDigitalTwinApp);
} else {
  initDigitalTwinApp();
}
