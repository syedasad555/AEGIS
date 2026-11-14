/**
 * AEGIS Evacuation Vision Console — Front-end Application Logic
 * ===========================================================================
 * Features:
 *   - Video-Selection-First Workflow with Preloaded Presets & Custom File Upload
 *   - Real-Time Live AI Stream (WebSocket) with YOLOv8n + ByteTrack Sensor Feed
 *   - Architectural 2D Digital Twin Virtual Map (Recognizable Corridor Layout)
 *   - Anti-Collision Pedestrian Label & Leader-Line Resolution
 *   - Anomaly rules: WRONG_DIRECTION, PROLONGED_STOPPING (corridor flow only)
 *   - Legend: blue=normal, red=confirmed anomaly, amber=selection
 *   - Interactive Alert Click-to-Inspect Synchronization across CCTV & Digital Twin
 *   - Defensible Severity Classification (CRITICAL / WARNING) & Duration Tracking
 *   - Responsive 100vh Desktop Layout (1366×768 / 1920×1080)
 */

class DigitalTwinApp {
  constructor() {
    // Initial Video Selection Screen elements
    this.videoSelectionScreen = document.getElementById('videoSelectionScreen');
    this.dropZone = document.getElementById('dropZone');
    this.dropzoneEmpty = document.getElementById('dropzoneEmpty');
    this.browseFilesBtn = document.getElementById('browseFilesBtn');
    this.fileDetailsCard = document.getElementById('fileDetailsCard');
    this.fileFormatBadge = document.getElementById('fileFormatBadge');
    this.fileSizeBadge = document.getElementById('fileSizeBadge');
    this.fileStatusIndicator = document.getElementById('fileStatusIndicator');
    this.removeFileBtn = document.getElementById('removeFileBtn');
    this.presetCards = document.querySelectorAll('.preset-card');
    this.videoFileInput = document.getElementById('videoFileInput');
    this.selectedFileName = document.getElementById('selectedFileName');
    this.customPathInput = document.getElementById('customPathInput');
    this.validatePathBtn = document.getElementById('validatePathBtn');
    this.initialDirectionSelect = document.getElementById('initialDirectionSelect');
    this.initialDirectionGroup = document.getElementById('initialDirectionGroup');
    this.selectionValidationMsg = document.getElementById('selectionValidationMsg');
    this.startAnalysisBtn = document.getElementById('startAnalysisBtn');
    this.activeVideoName = document.getElementById('activeVideoName');
    this.changeVideoBtn = document.getElementById('changeVideoBtn');

    // Digital Twin Filter State ('ALL' | 'ANOMALIES')
    this.twinFilter = 'ALL';
    this.twinFilterAllBtn = document.getElementById('twinFilterAll');
    this.twinFilterAnomaliesBtn = document.getElementById('twinFilterAnomalies');

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
    this.selectionPanel = document.getElementById('selectionPanel');
    this.selTrackId = document.getElementById('selTrackId');
    this.selPosition = document.getElementById('selPosition');
    this.selDirection = document.getElementById('selDirection');
    this.selZone = document.getElementById('selZone');
    this.selStatus = document.getElementById('selStatus');
    this.selAnomaly = document.getElementById('selAnomaly');

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

    // Only main evacuation corridor (normalized fallback for twin)
    this.corridorConfig = {
      name: 'MAIN EVACUATION CORRIDOR',
      points: [[0.06, 0.16], [0.94, 0.16], [0.94, 0.84], [0.06, 0.84]],
      flow_line: [[0.08, 0.50], [0.92, 0.50]],
      expected_direction: 'LEFT',
    };
    this.zonesConfig = { evacuation_route: [this.corridorConfig] };

    // Stream and rendering dimensions
    this.videoRefW = 640;
    this.videoRefH = 480;
    this.isPaused = false;
    this.ws = null;
    this.pulseAngle = 0;
    this.flowAnimOffset = 0;
    this.currentPedestrians = [];
    this.currentVideoName = null;
    this.mapZoom = 1.0;
    this.mapPanX = 0;
    this.mapPanY = 0;
    this._labelSlots = [];

    // Colour legend (must match backend visual_state)
    this.COLOR_NORMAL = '#38bdf8';
    this.COLOR_ANOMALY = '#ef4444';
    this.COLOR_SELECTED = '#f59e0b';
    this.COLOR_CORRIDOR = '#10b981';

    // Application state
    this.selectedVideoPath = 'assets/evacuation_test_corridor.avi';
    this.selectedVideoName = 'evacuation_test_corridor.avi';
    this.selectedDirection = 'LEFT';
    this.selectedSpeed = 1.0;
    this.selectedProfile = 'balanced';
    this.isAnalysisActive = false;
    this.diagnosticMode = false;

    // Playback Speed & Profile UI elements
    this.speedButtons = document.querySelectorAll('.speed-btn[data-speed]');
    this.customSpeedInput = document.getElementById('customSpeedInput');
    this.applyCustomSpeedBtn = document.getElementById('applyCustomSpeedBtn');
    this.requestedSpeedBadge = document.getElementById('requestedSpeedBadge');
    this.achievedSpeedBadge = document.getElementById('achievedSpeedBadge');
    this.monitoringModeBadge = document.getElementById('monitoringModeBadge');
    this.detectionProfileSelect = document.getElementById('detectionProfileSelect');

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

    // Playback Speed Controls
    if (this.speedButtons && this.speedButtons.length > 0) {
      this.speedButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
          const spd = parseFloat(btn.getAttribute('data-speed'));
          if (!isNaN(spd) && spd > 0) {
            this.setPlaybackSpeed(spd);
          }
        });
      });
    }

    if (this.applyCustomSpeedBtn && this.customSpeedInput) {
      const applyCustom = () => {
        const val = parseFloat(this.customSpeedInput.value);
        if (!isNaN(val) && val >= 0.1 && val <= 5.0) {
          this.setPlaybackSpeed(val);
        } else {
          this.showToast('Speed must be between 0.1x and 5.0x', 'error');
        }
      };
      this.applyCustomSpeedBtn.addEventListener('click', applyCustom);
      this.customSpeedInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          applyCustom();
        }
      });
    }

    // Detection Profile Selector
    if (this.detectionProfileSelect) {
      this.detectionProfileSelect.addEventListener('change', (e) => {
        const prof = e.target.value;
        this.setDetectionProfile(prof);
      });
    }

    // Corridor & Rules Modal Listeners
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

    // Virtual Map zoom (wheel) — denser labels when zoomed in
    if (this.canvas) {
      this.canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const factor = e.deltaY < 0 ? 1.08 : 0.92;
        this.mapZoom = Math.max(0.75, Math.min(3.5, this.mapZoom * factor));
        this.drawDigitalTwinMap(this.currentPedestrians, 1, null, this.selectedDirection);
      }, { passive: false });
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
    // Browse Files button triggers hidden file input
    if (this.browseFilesBtn && this.videoFileInput) {
      this.browseFilesBtn.addEventListener('click', () => {
        this.videoFileInput.click();
      });
    }

    // Drag and drop support on dropZone
    if (this.dropZone) {
      ['dragenter', 'dragover'].forEach((eventName) => {
        this.dropZone.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.dropZone.classList.add('drag-active');
        }, false);
      });

      ['dragleave', 'drop'].forEach((eventName) => {
        this.dropZone.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.dropZone.classList.remove('drag-active');
        }, false);
      });

      this.dropZone.addEventListener('drop', async (e) => {
        const dt = e.dataTransfer;
        const files = dt ? dt.files : null;
        if (files && files.length > 0) {
          await this.handleFileUpload(files[0]);
        }
      });
    }

    // Custom File Input
    if (this.videoFileInput) {
      this.videoFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        await this.handleFileUpload(file);
        this.videoFileInput.value = '';
      });
    }

    // Remove File button to reset selection
    if (this.removeFileBtn) {
      this.removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        this.resetFileSelection();
      });
    }

    // Initial Direction Buttons (LEFT, RIGHT, UP, DOWN)
    if (this.initialDirectionGroup) {
      const dirBtns = this.initialDirectionGroup.querySelectorAll('.dir-option-btn');
      dirBtns.forEach((btn) => {
        btn.addEventListener('click', () => {
          dirBtns.forEach((b) => b.classList.remove('active'));
          btn.classList.add('active');
          const d = this.coerceAdminDirection(btn.getAttribute('data-dir') || 'LEFT');
          this.selectedDirection = d;
          if (this.initialDirectionSelect) this.initialDirectionSelect.value = d;
          if (this.directionSelect) this.directionSelect.value = d;
        });
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
            const fname = data.filename || path.split(/[\/\\]/).pop();
            const fmt = (fname.split('.').pop() || 'VIDEO').toUpperCase();
            this.setFileSelected({
              filename: fname,
              path: data.path,
              format: fmt,
              sizeText: `${data.total_frames || data.frames || 0} frames (${data.width}×${data.height})`
            });
            this.showValidationMsg(`✓ Video verified: ${fname} (${data.width}×${data.height}, ${data.fps} FPS, ${data.total_frames || data.frames} frames)`, 'success');
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

    // Evacuation direction dropdown in initial screen (cardinals only)
    if (this.initialDirectionSelect) {
      this.initialDirectionSelect.addEventListener('change', (e) => {
        this.selectedDirection = this.coerceAdminDirection(e.target.value);
        e.target.value = this.selectedDirection;
        if (this.directionSelect) {
          this.directionSelect.value = this.selectedDirection;
        }
        if (this.initialDirectionGroup) {
          const dirBtns = this.initialDirectionGroup.querySelectorAll('.dir-option-btn');
          dirBtns.forEach((btn) => {
            btn.classList.toggle('active', btn.getAttribute('data-dir') === this.selectedDirection);
          });
        }
      });
    }

    // Start Analysis button
    if (this.startAnalysisBtn) {
      this.startAnalysisBtn.addEventListener('click', async () => {
        await this.startAnalysis();
      });
    }

    // Digital Twin Filter Toggles (ALL vs ANOMALIES ONLY)
    if (this.twinFilterAllBtn) {
      this.twinFilterAllBtn.addEventListener('click', () => {
        this.setTwinFilter('ALL');
      });
    }
    if (this.twinFilterAnomaliesBtn) {
      this.twinFilterAnomaliesBtn.addEventListener('click', () => {
        this.setTwinFilter('ANOMALIES');
      });
    }
  }

  setTwinFilter(mode) {
    this.twinFilter = (mode === 'ANOMALIES') ? 'ANOMALIES' : 'ALL';
    if (this.twinFilterAllBtn && this.twinFilterAnomaliesBtn) {
      if (this.twinFilter === 'ANOMALIES') {
        this.twinFilterAnomaliesBtn.classList.add('active');
        this.twinFilterAllBtn.classList.remove('active');
      } else {
        this.twinFilterAllBtn.classList.add('active');
        this.twinFilterAnomaliesBtn.classList.remove('active');
      }
    }
    // Redraw digital twin immediately with current pedestrians
    this.drawDigitalTwinMap(this.currentPedestrians, 1, null, this.selectedDirection);
  }

  async handleFileUpload(file) {
    if (!file) return;
    const fname = file.name;
    const fmt = (fname.split('.').pop() || '').toUpperCase();
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    this.showValidationMsg(`Uploading ${fname} (${sizeMb} MB)...`, 'info');

    try {
      const res = await fetch(`/api/upload_video?filename=${encodeURIComponent(fname)}`, {
        method: 'POST',
        body: file
      });
      const data = await res.json();

      if (data.status === 'ok' && data.valid) {
        this.setFileSelected({
          filename: data.filename || fname,
          path: data.path,
          format: fmt || 'VIDEO',
          sizeText: `${sizeMb} MB (${data.width}×${data.height})`
        });
        this.showValidationMsg(`✓ Video uploaded and verified: ${data.filename || fname} (${data.width}×${data.height}, ${data.fps} FPS)`, 'success');
      } else {
        this.showValidationMsg(`✕ Upload failed: ${data.error || 'Invalid video format'}`, 'error');
      }
    } catch (err) {
      this.showValidationMsg(`✕ Upload error: ${err.message}`, 'error');
    }
  }

  setFileSelected(meta) {
    this.selectedVideoPath = meta.path;
    this.selectedVideoName = meta.filename;

    if (this.selectedFileName) {
      this.selectedFileName.textContent = meta.filename;
    }
    if (this.fileFormatBadge) {
      this.fileFormatBadge.textContent = meta.format;
    }
    if (this.fileSizeBadge) {
      this.fileSizeBadge.textContent = meta.sizeText;
    }
    if (this.dropzoneEmpty) {
      this.dropzoneEmpty.style.display = 'none';
    }
    if (this.fileDetailsCard) {
      this.fileDetailsCard.style.display = 'block';
    }
    if (this.customPathInput) {
      this.customPathInput.value = meta.path;
    }
  }

  resetFileSelection() {
    this.selectedVideoPath = '';
    this.selectedVideoName = '';
    if (this.videoFileInput) this.videoFileInput.value = '';
    if (this.customPathInput) this.customPathInput.value = '';
    if (this.dropzoneEmpty) this.dropzoneEmpty.style.display = 'flex';
    if (this.fileDetailsCard) this.fileDetailsCard.style.display = 'none';
    this.hideValidationMsg();
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
        direction: this.selectedDirection,
        speed: this.selectedSpeed,
        profile: this.selectedProfile
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
            if (data.playback_speed !== undefined) {
              this.updateSpeedUI(parseFloat(data.playback_speed));
            }
            if (data.detection_profile && this.detectionProfileSelect) {
              this.detectionProfileSelect.value = data.detection_profile;
              this.selectedProfile = data.detection_profile;
            }
          } else if (data.type === 'speed_updated') {
            if (data.playback_speed !== undefined) {
              this.updateSpeedUI(parseFloat(data.playback_speed), parseFloat(data.achieved_playback_speed ?? data.playback_speed));
            }
          } else if (data.type === 'profile_updated') {
            if (data.detection_profile && this.detectionProfileSelect) {
              this.detectionProfileSelect.value = data.detection_profile;
              this.selectedProfile = data.detection_profile;
              this.showToast(`Detection Profile: ${data.detection_profile.toUpperCase()}`, 'info');
            }
          } else if (data.type === 'frame_update') {
            if (this.statusText) {
              const ver = data.classifier_version || data.runtime_id || '';
              this.statusText.textContent = ver ? `WS · ${ver}` : 'WS STREAM';
              this.statusText.title = data.runtime_id || ver || '';
            }
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

  coerceAdminDirection(direction) {
    const d = String(direction || 'LEFT').toUpperCase().replace(/-/g, '_');
    if (d === 'LEFT' || d === 'RIGHT' || d === 'UP' || d === 'DOWN') return d;
    if (d === 'WEST') return 'LEFT';
    if (d === 'EAST') return 'RIGHT';
    if (d === 'NORTH') return 'UP';
    if (d === 'SOUTH') return 'DOWN';
    return 'LEFT';
  }

  setPlaybackSpeed(speed) {
    const spd = Math.max(0.1, Math.min(5.0, parseFloat(speed)));
    if (isNaN(spd)) return;
    this.selectedSpeed = spd;
    this.updateSpeedUI(spd, spd);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        action: 'set_speed',
        speed: spd
      }));
    }
  }

  updateSpeedUI(requestedSpeed, achievedSpeed) {
    if (requestedSpeed !== undefined && requestedSpeed !== null) {
      this.selectedSpeed = requestedSpeed;
      if (this.requestedSpeedBadge) {
        this.requestedSpeedBadge.textContent = `Req: ${requestedSpeed}x`;
      }
      if (this.speedButtons) {
        this.speedButtons.forEach((btn) => {
          const bSpeed = parseFloat(btn.getAttribute('data-speed'));
          btn.classList.toggle('active', Math.abs(bSpeed - requestedSpeed) < 0.01);
        });
      }
      if (this.customSpeedInput) {
        const isPreset = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0].some((s) => Math.abs(s - requestedSpeed) < 0.01);
        if (!isPreset) {
          this.customSpeedInput.value = requestedSpeed;
        }
      }
    }
    if (achievedSpeed !== undefined && achievedSpeed !== null && this.achievedSpeedBadge) {
      this.achievedSpeedBadge.textContent = `Act: ${Number(achievedSpeed).toFixed(2)}x`;
      const isLagging = (this.selectedSpeed - achievedSpeed) > 0.25;
      this.achievedSpeedBadge.classList.toggle('lagging', isLagging);
    }
  }

  setDetectionProfile(profile) {
    const prof = String(profile).toLowerCase();
    this.selectedProfile = prof;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        action: 'set_detection_profile',
        profile: prof
      }));
    }
  }

  switchDirection(direction) {
    const dir = this.coerceAdminDirection(direction);
    this.selectedDirection = dir;
    // Live update — do NOT restart the stream / tracker
    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.isAnalysisActive) {
      this.ws.send(JSON.stringify({
        action: 'set_direction',
        direction: dir
      }));
    }
    this.drawDigitalTwinMap(this.currentPedestrians, 1, null, dir);
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

    this.updateSelectionPanel();

    // Redraw Digital Twin canvas immediately
    this.drawDigitalTwinMap(this.currentPedestrians, 1, null, this.selectedDirection);
  }

  updateSelectionPanel() {
    if (!this.selectionPanel) return;
    if (this.selectedTrackId === null) {
      this.selectionPanel.style.display = 'none';
      return;
    }
    const ped = (this.currentPedestrians || []).find((p) => p.track_id === this.selectedTrackId);
    this.selectionPanel.style.display = 'block';
    if (!ped) {
      if (this.selTrackId) this.selTrackId.textContent = `#${this.selectedTrackId}`;
      if (this.selStatus) this.selStatus.textContent = 'not in frame';
      return;
    }
    const foot = ped.foot || [0, 0];
    if (this.selTrackId) this.selTrackId.textContent = `#${ped.track_id}`;
    if (this.selPosition) this.selPosition.textContent = `${Math.round(foot[0])}, ${Math.round(foot[1])}`;
    if (this.selDirection) this.selDirection.textContent = ped.direction || 'UNKNOWN';
    if (this.selZone) {
      this.selZone.textContent = ped.in_corridor
        ? (ped.zone_name || 'Corridor')
        : (ped.zone_name || 'Outside corridor');
    }
    if (this.selStatus) {
      this.selStatus.textContent = ped.movement_status || (ped.anomaly ? 'ALERT' : 'OK');
    }
    if (this.selAnomaly) {
      this.selAnomaly.textContent = ped.anomaly
        ? this.formatAnomalyLabel(ped.anomaly_type)
        : (ped.direction_classification === 'UNKNOWN' ? 'Unknown dir' : 'None');
    }
    // Direction diagnostic block (selected pedestrian only)
    let diagEl = document.getElementById('selDiagnostics');
    if (!diagEl && this.selectionPanel) {
      diagEl = document.createElement('pre');
      diagEl.id = 'selDiagnostics';
      diagEl.className = 'sel-diagnostics';
      this.selectionPanel.appendChild(diagEl);
    }
    if (diagEl) {
      const d = ped.direction_diagnostics || {
        estimated_direction: ped.direction,
        expected_direction: ped.expected_direction,
        sector_distance: ped.sector_distance,
        angular_deviation_deg: ped.angular_deviation_deg,
        classification: ped.direction_classification,
        direction_confidence: ped.direction_confidence,
        reason: ped.direction_reason,
        final_visual_state: ped.visual_state,
        dx: (ped.direction_vector || [])[0],
        dy: (ped.direction_vector || [])[1],
        allowed_directions: ped.allowed_directions,
      };
      const dx = d.dx ?? (d.direction_vector || [])[0];
      const dy = d.dy ?? (d.direction_vector || [])[1];
      diagEl.textContent = [
        `id=${d.track_id ?? ped.track_id ?? ped.id}  dx=${dx ?? '—'}  dy=${dy ?? '—'}`,
        `est=${d.estimated_direction || ped.direction}  conf=${(d.direction_confidence ?? ped.direction_confidence ?? 0).toFixed?.(2) ?? d.direction_confidence}`,
        `exp=${d.expected_direction || '—'}  sectorΔ=${d.sector_distance ?? ped.sector_distance ?? '—'}  allowed=${(d.allowed_directions || ped.allowed_directions || []).join(',') || '—'}`,
        `class=${d.classification || ped.direction_classification || '—'}  state=${d.final_visual_state || ped.visual_state}`,
        d.reason || ped.direction_reason || '',
      ].join('\n');
    }
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
    this.updateSelectionPanel();

    // Sync playback speed and profile if reflected in payload
    if (data.playback_speed !== undefined) {
      this.updateSpeedUI(parseFloat(data.requested_playback_speed ?? data.playback_speed), parseFloat(data.achieved_playback_speed ?? data.playback_speed));
    }
    if (data.detection_profile && data.detection_profile !== this.selectedProfile && this.detectionProfileSelect) {
      this.selectedProfile = data.detection_profile;
      this.detectionProfileSelect.value = data.detection_profile;
    }
    if (this.monitoringModeBadge && data.monitoring_mode) {
      this.monitoringModeBadge.textContent = `Mode: ${data.monitoring_mode}`;
      this.monitoringModeBadge.className = `speed-badge mode-badge ${String(data.monitoring_mode).toLowerCase()}`;
    }

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
      PROLONGED_STOPPING: 'Prolonged Stopping',
      WRONG_DIRECTION: 'Wrong Direction Flow',
    };
    return map[type] || type.replace(/_/g, ' ');
  }

  isConfirmedAnomaly(p) {
    if (!p) return false;
    if (p.visual_state === 'anomaly') return true;
    return !!(p.anomaly && (p.anomaly_type === 'WRONG_DIRECTION' || p.anomaly_type === 'PROLONGED_STOPPING'));
  }

  markerColor(p, isSelected) {
    // Backend visual_state is authoritative. UNKNOWN → blue (not red).
    // Selection never overwrites fill — amber is outline only.
    if (this.isConfirmedAnomaly(p)) return this.COLOR_ANOMALY;
    return this.COLOR_NORMAL;
  }

  placeMapLabel(anchorX, anchorY, textW, textH, occupied) {
    const candidates = [
      [8, -textH - 4], [-(textW + 8), -textH - 4], [8, 8],
      [-(textW + 8), 8], [0, -textH - 18], [0, 16],
      [textW + 12, 0], [-(textW + 12), 0],
    ];
    for (const [ox, oy] of candidates) {
      const lx = anchorX + ox;
      const ly = anchorY + oy;
      const rect = { x1: lx, y1: ly, x2: lx + textW, y2: ly + textH };
      let hit = false;
      for (const o of occupied) {
        if (!(rect.x2 < o.x1 || rect.x1 > o.x2 || rect.y2 < o.y1 || rect.y1 > o.y2)) {
          hit = true;
          break;
        }
      }
      if (!hit) {
        occupied.push(rect);
        return { lx, ly, leader: Math.abs(ox) > 10 || Math.abs(oy) > textH + 6 };
      }
    }
    occupied.push({ x1: anchorX + 8, y1: anchorY - textH - 4, x2: anchorX + 8 + textW, y2: anchorY - 4 });
    return { lx: anchorX + 8, ly: anchorY - textH - 4, leader: false };
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
      WRONG_DIRECTION: 4,
      PROLONGED_STOPPING: 2
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
      const severity = a.severity || 'WARNING';
      const sevClass = (severity === 'CRITICAL') ? 'critical' : 'warning';
      const duration = a.duration_sec ? `${a.duration_sec}s` : (a.stopping_duration_sec ? `${a.stopping_duration_sec}s` : 'live');
      const ts = this.formatTimestamp(a.timestamp);
      const typeLabel = this.formatAnomalyLabel(atype);
      const zoneHint = a.evidence?.reason || a.zone_name || '';

      html += `
        <div class="incident-item ${sevClass} ${isSel ? 'selected' : ''}" data-track-id="${tid}">
          <div class="incident-id">#${tid}</div>
          <div class="incident-main">
            <div class="incident-type">${typeLabel}</div>
            <div class="incident-meta">${ts} · ${duration}${zoneHint ? ' · ' + zoneHint : ''}</div>
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

    // 4. Main evacuation corridor ONLY (subtle boundary + green flow)
    const corridor = this.corridorConfig || (this.zonesConfig.evacuation_route || [])[0];
    if (corridor) {
      this.drawArchitecturalZone(
        ctx, padX, padY, cw, ch, corridor,
        'rgba(16, 185, 129, 0.04)', this.COLOR_CORRIDOR, ''
      );
    }
    this.drawCorridorFlowChevrons(ctx, padX, padY, cw, ch, expectedDirection);

    // Colour legend
    this.drawMapLegendAndFootnote(ctx, w, h);

    // Presentation Filter: in ANOMALIES mode, display only pedestrians whose backend visual_state === 'anomaly'
    const activePedestrians = (this.twinFilter === 'ANOMALIES')
      ? (pedestrians || []).filter(p => p && p.visual_state === 'anomaly')
      : (pedestrians || []);

    if (this.twinFilter === 'ANOMALIES') {
      ctx.save();
      ctx.font = '600 11px var(--font-mono, monospace)';
      ctx.fillStyle = activePedestrians.length > 0 ? '#ef4444' : '#64748b';
      ctx.textAlign = 'right';
      ctx.fillText(`FILTER: ANOMALIES ONLY (${activePedestrians.length} ACTIVE)`, w - 16, 24);
      ctx.restore();
    }

    if (!activePedestrians || activePedestrians.length === 0) return;

    // 7. Trajectories first (below pedestrian marker dots)
    const videoRefW = this.videoRefW || 640;
    const videoRefH = this.videoRefH || 480;
    const maxDisplacementPx = 65; // px in video space (aligned with config trajectory_max_displacement_px)
    const maxFrameGap = 10;       // aligned with config trajectory_max_frame_gap

    // Apply zoom transform for dense crowds
    ctx.save();
    ctx.translate(w / 2 + this.mapPanX, h / 2 + this.mapPanY);
    ctx.scale(this.mapZoom, this.mapZoom);
    ctx.translate(-w / 2, -h / 2);

    for (const p of activePedestrians) {
      const isAnomaly = this.isConfirmedAnomaly(p);
      const isSelected = (this.selectedTrackId !== null && p.track_id === this.selectedTrackId);
      const trail = p.trajectory;

      // Trails only for selected / anomalous
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
    const sortedPedestrians = [...activePedestrians].sort((a, b) => {
      const ya = (a.bbox ? a.bbox[1] : (a.y || 0));
      const yb = (b.bbox ? b.bbox[1] : (b.y || 0));
      return ya - yb;
    });

    // Zoom-dependent label density: show all markers; sparse labels when zoomed out
    const showAllLabels = this.mapZoom >= 1.35 || activePedestrians.length <= 40;
    const labelPriority = [...sortedPedestrians].sort((a, b) => {
      const sa = this.isConfirmedAnomaly(a) || a.track_id === this.selectedTrackId ? 1 : 0;
      const sb = this.isConfirmedAnomaly(b) || b.track_id === this.selectedTrackId ? 1 : 0;
      return sb - sa;
    });
    const maxLabels = showAllLabels ? activePedestrians.length : Math.min(24, activePedestrians.length);
    const labelIds = new Set(labelPriority.slice(0, maxLabels).map((p) => p.track_id));

    for (const p of sortedPedestrians) {
      const tid = p.track_id;
      const isAnomaly = this.isConfirmedAnomaly(p);
      const isSelected = (this.selectedTrackId !== null && tid === this.selectedTrackId);

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

      // Fill from anomaly state only — selection uses amber outline
      const dotColor = this.markerColor(p, isSelected);

      if (isAnomaly) {
        const pulseRadius = 10 + Math.sin(this.pulseAngle) * 3;
        ctx.beginPath();
        ctx.arc(mapX, mapY, pulseRadius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      if (isSelected) {
        const selRadius = 13 + Math.sin(this.pulseAngle * 1.5) * 2;
        ctx.beginPath();
        ctx.arc(mapX, mapY, selRadius, 0, Math.PI * 2);
        ctx.strokeStyle = this.COLOR_SELECTED;
        ctx.lineWidth = 2;
        ctx.stroke();
        const bSize = 6;
        ctx.strokeStyle = this.COLOR_SELECTED;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(mapX - selRadius, mapY - selRadius + bSize); ctx.lineTo(mapX - selRadius, mapY - selRadius); ctx.lineTo(mapX - selRadius + bSize, mapY - selRadius); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(mapX + selRadius - bSize, mapY - selRadius); ctx.lineTo(mapX + selRadius, mapY - selRadius); ctx.lineTo(mapX + selRadius, mapY - selRadius + bSize); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(mapX - selRadius, mapY + selRadius - bSize); ctx.lineTo(mapX - selRadius, mapY + selRadius); ctx.lineTo(mapX - selRadius + bSize, mapY + selRadius); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(mapX + selRadius - bSize, mapY + selRadius); ctx.lineTo(mapX + selRadius, mapY + selRadius); ctx.lineTo(mapX + selRadius, mapY + selRadius - bSize); ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(mapX, mapY, isSelected || isAnomaly ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = dotColor;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.0;
      ctx.stroke();

      if (!labelIds.has(tid) && !isSelected && !isAnomaly) continue;

      let tagText = `#${tid}`;
      if (isAnomaly) {
        const shortMap = { PROLONGED_STOPPING: 'STOP', WRONG_DIRECTION: 'DIR' };
        tagText = `#${tid} · ${shortMap[p.anomaly_type] || 'ALRT'}`;
      }

      ctx.font = '600 9.5px sans-serif';
      const tw = ctx.measureText(tagText).width;
      const tagW = tw + 8;
      const tagH = 16;
      const placed = this.placeMapLabel(mapX, mapY, tagW, tagH, occupiedLabelRects);
      const chosen = {
        x1: placed.lx, y1: placed.ly, x2: placed.lx + tagW, y2: placed.ly + tagH,
        needLeader: placed.leader,
      };

      if (chosen.needLeader) {
        ctx.beginPath();
        ctx.moveTo(mapX, mapY);
        ctx.lineTo(chosen.x1 + tagW / 2, chosen.y1 + tagH / 2);
        ctx.strokeStyle = isSelected
          ? 'rgba(245, 158, 11, 0.7)'
          : (isAnomaly ? 'rgba(239, 68, 68, 0.45)' : 'rgba(56, 189, 248, 0.35)');
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(10, 15, 26, 0.92)';
      ctx.fillRect(chosen.x1, chosen.y1, tagW, tagH);
      ctx.strokeStyle = isSelected ? this.COLOR_SELECTED : (isAnomaly ? this.COLOR_ANOMALY : 'rgba(56, 189, 248, 0.5)');
      ctx.lineWidth = isSelected ? 1.5 : 1;
      ctx.strokeRect(chosen.x1, chosen.y1, tagW, tagH);
      ctx.fillStyle = isAnomaly ? '#fca5a5' : '#f8fafc';
      ctx.fillText(tagText, chosen.x1 + 4, chosen.y1 + 11.5);
    }
    ctx.restore();
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

      const dir = String(direction || 'EAST').toUpperCase();
      const goWest = dir === 'LEFT' || dir === 'WEST';
      const goEast = dir === 'RIGHT' || dir === 'EAST';
      const goNorth = dir === 'UP' || dir === 'NORTH';
      const goSouth = dir === 'DOWN' || dir === 'SOUTH';

      if (goWest) {
        cx -= this.flowAnimOffset;
        if (cx < padX + 20) cx += cw - 40;
        ctx.beginPath();
        ctx.moveTo(cx + 6, cy - 5);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + 6, cy + 5);
        ctx.stroke();
      } else if (goEast || (!goNorth && !goSouth)) {
        cx += this.flowAnimOffset;
        if (cx > padX + cw - 20) cx -= cw - 40;
        ctx.beginPath();
        ctx.moveTo(cx - 6, cy - 5);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx - 6, cy + 5);
        ctx.stroke();
      } else if (goNorth) {
        cy -= this.flowAnimOffset;
        if (cy < padY + 20) cy += ch - 40;
        ctx.beginPath();
        ctx.moveTo(cx - 5, cy + 6);
        ctx.lineTo(cx, cy);
        ctx.lineTo(cx + 5, cy + 6);
        ctx.stroke();
      } else if (goSouth) {
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
    const ly = h - 28;
    ctx.fillStyle = 'rgba(10, 15, 26, 0.82)';
    ctx.fillRect(lx - 6, ly - 10, 268, 22);

    ctx.font = '500 9px sans-serif';
    // Blue = normal
    ctx.fillStyle = this.COLOR_NORMAL;
    ctx.beginPath(); ctx.arc(lx + 4, ly, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Normal', lx + 12, ly + 3);
    // Red = anomaly
    ctx.fillStyle = this.COLOR_ANOMALY;
    ctx.beginPath(); ctx.arc(lx + 68, ly, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Anomaly', lx + 76, ly + 3);
    // Amber = selected
    ctx.strokeStyle = this.COLOR_SELECTED;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(lx + 140, ly, 4, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Selected', lx + 150, ly + 3);
    // Green = corridor flow
    ctx.fillStyle = this.COLOR_CORRIDOR;
    ctx.fillText('Flow →', lx + 210, ly + 3);

    ctx.fillStyle = 'rgba(148, 163, 184, 0.55)';
    ctx.font = '500 8px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(`Zoom ${this.mapZoom.toFixed(2)}× · scroll to zoom`, w - 28, h - 10);
    ctx.textAlign = 'left';
  }

  /* ==========================================================================
     Modal & Config Management
     ========================================================================== */

  openZonesModal() {
    if (this.zonesModal) {
      this.zonesModal.classList.add('active');
      if (this.cfgZonesJson) {
        this.cfgZonesJson.value = JSON.stringify({
          evacuation_corridor: this.corridorConfig,
        }, null, 2);
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
      const parsed = JSON.parse(this.cfgZonesJson.value);
      const corr = parsed.evacuation_corridor || parsed;
      this.corridorConfig = corr;
      this.zonesConfig = { evacuation_route: [corr] };

      const payload = {
        evacuation_corridor: corr,
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
        this.showToast('Evacuation corridor & thresholds updated', 'success');
        this.closeZonesModal();
      } else {
        const err = await res.json().catch(() => ({}));
        this.showToast(err.message || 'Failed to save corridor config', 'error');
      }
    } catch (e) {
      this.showToast('Invalid JSON format: ' + e.message, 'error');
    }
  }

  resetZonesToDefault() {
    this.corridorConfig = {
      name: 'MAIN EVACUATION CORRIDOR',
      points: [[0.06, 0.16], [0.94, 0.16], [0.94, 0.84], [0.06, 0.84]],
      flow_line: [[0.08, 0.50], [0.92, 0.50]],
      expected_direction: 'LEFT',
    };
    this.zonesConfig = { evacuation_route: [this.corridorConfig] };
    if (this.cfgZonesJson) {
      this.cfgZonesJson.value = JSON.stringify({ evacuation_corridor: this.corridorConfig }, null, 2);
    }
    if (this.cfgSpeedThreshold) this.cfgSpeedThreshold.value = '0.6';
    if (this.cfgStoppingFrames) this.cfgStoppingFrames.value = '30';
    this.showToast('Reset to default evacuation corridor', 'info');
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


// =========================================================================
// Global Debug Panel (Ctrl+Shift+D) — shows per-track pipeline state
// =========================================================================
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.shiftKey && (e.key === 'D' || e.key === 'd')) {
    e.preventDefault();
    const panel = document.getElementById('debugPanel');
    if (!panel) return;
    panel.style.display = (panel.style.display === 'none') ? 'block' : 'none';
    if (panel.style.display === 'block' && window.app) {
      window.app._updateDebugPanel();
    }
  }
});

// Add _updateDebugPanel to the DigitalTwinApp prototype
DigitalTwinApp.prototype._updateDebugPanel = function () {
  const el = document.getElementById('debugPanelContent');
  if (!el) return;
  const app = this;
  const lines = [];
  lines.push(`RUNTIME: ${app._lastRuntimeId || '—'}  WS: ${app.ws ? (app.ws.readyState === 1 ? 'OPEN' : 'CLOSED') : 'NONE'}`);
  lines.push(`EXPECTED_DIR: ${app.selectedDirection}  CLASSIFIER: ${app._lastClassifierVersion || '—'}`);
  lines.push(`DETECTION_PROFILE: ${(app._lastDetectionProfile || app.selectedProfile || 'balanced').toUpperCase()}  SPEED (Req/Act): ${app._lastPlaybackSpeed || app.selectedSpeed || 1.0}x / ${(app._lastAchievedSpeed ?? app.selectedSpeed ?? 1.0).toFixed(2)}x`);
  lines.push(`TRACKS: ${(app.currentPedestrians || []).length}  ANOMALIES: ${(app.currentPedestrians || []).filter(p => app.isConfirmedAnomaly(p)).length}`);
  lines.push(`CONF_THRESH: ${app._lastConfThresh ?? '—'}  IMGSZ: ${app._lastInferenceImgsz ?? '—'}  INFER_MS: ${app._lastInferMs ?? '—'}  EFF_FPS: ${app._lastEffectiveFps ?? '—'}`);
  lines.push(`MODE: ${app._lastMonitoringMode || 'NORMAL'}  SKIPPED: ${app._lastFramesSkipped || 0}  PROC_FPS: ${app._lastProcFps ?? '—'}  POLICY_MAX_SKIP: ${(app._lastAdaptivePolicy || {}).max_skip_allowed ?? '—'}`);
  lines.push('─'.repeat(50));
  const peds = (app.currentPedestrians || []).slice(0, 10);
  for (const p of peds) {
    const vs = p.visual_state || 'unknown';
    const anom = app.isConfirmedAnomaly(p);
    const diag = p.direction_diagnostics || {};
    lines.push(
      `#${p.track_id}  dir=${p.direction || '—'}  cls=${p.direction_classification || '—'}` +
      `  sectorΔ=${p.sector_distance ?? '—'}  vs=${vs}  ${anom ? '🔴 ANOMALY' : '🔵 normal'}`
    );
    if (diag.reason) {
      lines.push(`  reason: ${String(diag.reason).slice(0, 60)}`);
    }
  }
  el.textContent = lines.join('\n');
  // Auto-refresh while panel is open
  if (document.getElementById('debugPanel').style.display === 'block') {
    clearTimeout(app._debugRefreshTimer);
    app._debugRefreshTimer = setTimeout(() => app._updateDebugPanel(), 300);
  }
};

// Hook into renderLiveFrame to capture runtime metadata for the debug panel
const _origRenderLiveFrame = DigitalTwinApp.prototype.renderLiveFrame;
DigitalTwinApp.prototype.renderLiveFrame = function (data) {
  if (data.runtime_id) this._lastRuntimeId = data.runtime_id;
  if (data.classifier_version) this._lastClassifierVersion = data.classifier_version;
  if (data.detection_profile) this._lastDetectionProfile = data.detection_profile;
  if (data.playback_speed !== undefined) this._lastPlaybackSpeed = data.playback_speed;
  if (data.achieved_playback_speed !== undefined) this._lastAchievedSpeed = data.achieved_playback_speed;
  if (data.effective_playback_fps !== undefined) this._lastEffectiveFps = data.effective_playback_fps;
  if (data.detection_confidence !== undefined) this._lastConfThresh = data.detection_confidence;
  if (data.inference_imgsz !== undefined) this._lastInferenceImgsz = data.inference_imgsz;
  if (data.infer_ms !== undefined) this._lastInferMs = data.infer_ms;
  if (data.processing_fps !== undefined) this._lastProcFps = data.processing_fps;
  if (data.frames_skipped !== undefined) this._lastFramesSkipped = data.frames_skipped;
  if (data.monitoring_mode) this._lastMonitoringMode = data.monitoring_mode;
  if (data.adaptive_policy) this._lastAdaptivePolicy = data.adaptive_policy;
  _origRenderLiveFrame.call(this, data);
};
