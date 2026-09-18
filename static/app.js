/**
 * VisionX Studio - Parallel Video Comparison & Synchronized Engine
 * Domain: Player Movement Heatmap Generation (VisionX 2026 PR-01)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const videoSelect = document.getElementById("videoSelect");
  const modeTrackBtn = document.getElementById("modeTrackBtn");
  const modeDetectBtn = document.getElementById("modeDetectBtn");
  const len150Btn = document.getElementById("len150Btn");
  const lenAllBtn = document.getElementById("lenAllBtn");
  const runPipelineBtn = document.getElementById("runPipelineBtn");
  const runBtnText = document.getElementById("runBtnText");

  const jobProgressContainer = document.getElementById("jobProgressContainer");
  const jobProgressBar = document.getElementById("jobProgressBar");
  const jobStageText = document.getElementById("jobStageText");
  const jobTelemetryText = document.getElementById("jobTelemetryText");
  const jobPercentText = document.getElementById("jobPercentText");

  const sourceVideo = document.getElementById("sourceVideo");
  const outputVideo = document.getElementById("outputVideo");
  const sourcePlaceholder = document.getElementById("sourcePlaceholder");
  const outputPlaceholder = document.getElementById("outputPlaceholder");
  const outputPlaceholderText = document.getElementById("outputPlaceholderText");
  const sourceMetaLabel = document.getElementById("sourceMetaLabel");
  const outputMetaLabel = document.getElementById("outputMetaLabel");
  const outputBadgeTag = document.getElementById("outputBadgeTag");

  const masterPlayBtn = document.getElementById("masterPlayBtn");
  const playIcon = document.getElementById("playIcon");
  const pauseIcon = document.getElementById("pauseIcon");
  const stepBackBtn = document.getElementById("stepBackBtn");
  const stepForwardBtn = document.getElementById("stepForwardBtn");
  const timelineScrubber = document.getElementById("timelineScrubber");
  const currentTimeDisplay = document.getElementById("currentTimeDisplay");
  const durationDisplay = document.getElementById("durationDisplay");
  const frameCounter = document.getElementById("frameCounter");
  const syncLockBtn = document.getElementById("syncLockBtn");
  const speedSelect = document.getElementById("speedSelect");
  const loopToggleBtn = document.getElementById("loopToggleBtn");

  const tabHeatmapsBtn = document.getElementById("tabHeatmapsBtn");
  const tabLogsBtn = document.getElementById("tabLogsBtn");
  const heatmapsPanel = document.getElementById("heatmapsPanel");
  const logsPanel = document.getElementById("logsPanel");
  const heatmapsGrid = document.getElementById("heatmapsGrid");
  const consoleLogs = document.getElementById("consoleLogs");

  const imageModal = document.getElementById("imageModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalImage = document.getElementById("modalImage");
  const modalCloseBtn = document.getElementById("modalCloseBtn");

  // State
  let videosList = [];
  let currentVideo = null;
  let selectedMode = "track"; // "track" | "detect"
  let selectedLength = 150; // 150 for 5s preview, 0 for all
  let syncLocked = true;
  let isLooping = false;
  let isScrubbing = false;
  let activeJobId = null;
  let pollInterval = null;
  let videoFps = 30.0;

  // ── Format Helpers ────────────────────────────────────────────────────────
  function formatTime(seconds) {
    if (isNaN(seconds) || seconds < 0) return "00:00.00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 100);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;
  }

  // ── Initial Video Fetch ───────────────────────────────────────────────────
  async function loadVideos() {
    try {
      const res = await fetch("/api/videos");
      const data = await res.json();
      videosList = data.videos || [];
      renderVideoDropdown();
      if (videosList.length > 0) {
        selectVideo(videosList[0].filename);
      }
    } catch (err) {
      console.error("Failed to load videos:", err);
      videoSelect.innerHTML = `<option value="" disabled>Error loading videos</option>`;
    }
  }

  function renderVideoDropdown() {
    videoSelect.innerHTML = "";
    videosList.forEach((v) => {
      const opt = document.createElement("option");
      opt.value = v.filename;
      const m = v.metadata;
      const resStr = m.width && m.height ? `${m.width}×${m.height}` : "";
      const durStr = m.duration ? `${m.duration}s` : "";
      opt.textContent = `${v.filename} (${durStr} ${resStr})`;
      videoSelect.appendChild(opt);
    });
  }

  function selectVideo(filename) {
    currentVideo = videosList.find((v) => v.filename === filename);
    if (!currentVideo) return;

    videoSelect.value = filename;
    const m = currentVideo.metadata;
    videoFps = m.fps || 30.0;

    sourceMetaLabel.textContent = `${m.width}×${m.height} • ${m.fps} FPS • ${m.duration}s`;
    sourceVideo.src = `/video/source/${encodeURIComponent(filename)}`;
    sourcePlaceholder.classList.add("hidden");

    // Load output video if existing
    loadOutputForCurrentVideo();
    loadHeatmaps();
  }

  function loadOutputForCurrentVideo() {
    if (!currentVideo) return;
    const outputs = currentVideo.outputs;

    let targetPreview = null;
    if (selectedMode === "track" && outputs.tracks_preview_file) {
      targetPreview = outputs.tracks_preview_file;
      outputBadgeTag.textContent = "VISIONX // BYTETRACK OUTPUT";
    } else if (selectedMode === "detect" && outputs.detect_preview_file) {
      targetPreview = outputs.detect_preview_file;
      outputBadgeTag.textContent = "VISIONX // DETECTION OUTPUT";
    } else if (outputs.tracks_preview_file) {
      targetPreview = outputs.tracks_preview_file;
      outputBadgeTag.textContent = "VISIONX // BYTETRACK OUTPUT";
    } else if (outputs.detect_preview_file) {
      targetPreview = outputs.detect_preview_file;
      outputBadgeTag.textContent = "VISIONX // DETECTION OUTPUT";
    }

    if (targetPreview) {
      outputVideo.src = `/video/output/${encodeURIComponent(targetPreview)}`;
      outputPlaceholder.classList.add("hidden");
      outputMetaLabel.textContent = "Ready • Processed Preview";
    } else {
      outputVideo.removeAttribute("src");
      outputVideo.load();
      outputPlaceholder.classList.remove("hidden");
      outputPlaceholderText.textContent = `No ${selectedMode} output yet. Click 'Run Pipeline' to generate.`;
      outputMetaLabel.textContent = "Awaiting Processing";
    }
  }

  // ── Mode & Length Selectors ───────────────────────────────────────────────
  modeTrackBtn.addEventListener("click", () => {
    selectedMode = "track";
    modeTrackBtn.classList.add("active");
    modeDetectBtn.classList.remove("active");
    loadOutputForCurrentVideo();
  });

  modeDetectBtn.addEventListener("click", () => {
    selectedMode = "detect";
    modeDetectBtn.classList.add("active");
    modeTrackBtn.classList.remove("active");
    loadOutputForCurrentVideo();
  });

  len150Btn.addEventListener("click", () => {
    selectedLength = 150;
    len150Btn.classList.add("active");
    lenAllBtn.classList.remove("active");
  });

  lenAllBtn.addEventListener("click", () => {
    selectedLength = 0;
    lenAllBtn.classList.add("active");
    len150Btn.classList.remove("active");
  });

  videoSelect.addEventListener("change", (e) => {
    selectVideo(e.target.value);
  });

  // ── Synchronized Playback Engine ─────────────────────────────────────────
  function togglePlayPause() {
    if (sourceVideo.paused) {
      sourceVideo.play();
      if (syncLocked && outputVideo.src) {
        outputVideo.currentTime = sourceVideo.currentTime;
        outputVideo.play();
      }
      playIcon.classList.add("hidden");
      pauseIcon.classList.remove("hidden");
    } else {
      sourceVideo.pause();
      if (outputVideo.src) {
        outputVideo.pause();
      }
      playIcon.classList.remove("hidden");
      pauseIcon.classList.add("hidden");
    }
  }

  masterPlayBtn.addEventListener("click", togglePlayPause);

  function syncDrift() {
    if (!syncLocked || !outputVideo.src || outputVideo.paused || sourceVideo.paused) return;
    const diff = outputVideo.currentTime - sourceVideo.currentTime;
    if (Math.abs(diff) > 0.04) {
      outputVideo.currentTime = sourceVideo.currentTime;
    }
  }

  sourceVideo.addEventListener("timeupdate", () => {
    if (!isScrubbing && sourceVideo.duration) {
      const pct = (sourceVideo.currentTime / sourceVideo.duration) * 100;
      timelineScrubber.value = pct;
    }

    currentTimeDisplay.textContent = formatTime(sourceVideo.currentTime);
    const frameIndex = Math.floor(sourceVideo.currentTime * videoFps);
    frameCounter.textContent = `Frame ${frameIndex}`;

    syncDrift();
  });

  sourceVideo.addEventListener("loadedmetadata", () => {
    durationDisplay.textContent = formatTime(sourceVideo.duration);
    timelineScrubber.value = 0;
  });

  sourceVideo.addEventListener("ended", () => {
    if (isLooping) {
      sourceVideo.currentTime = 0;
      sourceVideo.play();
      if (syncLocked && outputVideo.src) {
        outputVideo.currentTime = 0;
        outputVideo.play();
      }
    } else {
      playIcon.classList.remove("hidden");
      pauseIcon.classList.add("hidden");
    }
  });

  timelineScrubber.addEventListener("input", () => {
    isScrubbing = true;
    if (sourceVideo.duration) {
      const targetTime = (timelineScrubber.value / 100) * sourceVideo.duration;
      sourceVideo.currentTime = targetTime;
      if (syncLocked && outputVideo.src) {
        outputVideo.currentTime = targetTime;
      }
      currentTimeDisplay.textContent = formatTime(targetTime);
      frameCounter.textContent = `Frame ${Math.floor(targetTime * videoFps)}`;
    }
  });

  timelineScrubber.addEventListener("change", () => {
    isScrubbing = false;
  });

  function stepFrame(framesDelta) {
    sourceVideo.pause();
    if (outputVideo.src) outputVideo.pause();
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");

    const frameDuration = 1.0 / videoFps;
    const newTime = Math.max(0, Math.min(sourceVideo.duration || 0, sourceVideo.currentTime + framesDelta * frameDuration));
    sourceVideo.currentTime = newTime;
    if (syncLocked && outputVideo.src) {
      outputVideo.currentTime = newTime;
    }
  }

  stepBackBtn.addEventListener("click", () => stepFrame(-1));
  stepForwardBtn.addEventListener("click", () => stepFrame(1));

  syncLockBtn.addEventListener("click", () => {
    syncLocked = !syncLocked;
    syncLockBtn.classList.toggle("active", syncLocked);
    if (syncLocked && outputVideo.src) {
      outputVideo.currentTime = sourceVideo.currentTime;
      if (!sourceVideo.paused) outputVideo.play();
    }
  });

  speedSelect.addEventListener("change", (e) => {
    const rate = parseFloat(e.target.value);
    sourceVideo.playbackRate = rate;
    if (outputVideo.src) outputVideo.playbackRate = rate;
  });

  loopToggleBtn.addEventListener("click", () => {
    isLooping = !isLooping;
    loopToggleBtn.classList.toggle("active", isLooping);
  });

  // ── Keyboard Shortcuts ────────────────────────────────────────────────────
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA") {
      return;
    }
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === "ArrowLeft") {
      e.preventDefault();
      stepFrame(-1);
    } else if (e.code === "ArrowRight") {
      e.preventDefault();
      stepFrame(1);
    } else if (e.key === "s" || e.key === "S") {
      syncLockBtn.click();
    } else if (e.key === "l" || e.key === "L") {
      loopToggleBtn.click();
    } else if (e.key === "Escape") {
      closeModal();
    }
  });

  // ── Pipeline Runner ───────────────────────────────────────────────────────
  runPipelineBtn.addEventListener("click", async () => {
    if (!currentVideo) return;

    runPipelineBtn.disabled = true;
    runBtnText.textContent = "Running...";
    jobProgressContainer.classList.remove("hidden");
    jobProgressBar.style.width = "0%";
    jobPercentText.textContent = "0%";
    jobStageText.textContent = `Initializing ${selectedMode} pipeline...`;
    jobTelemetryText.textContent = "Starting...";

    // Switch to logs tab automatically
    tabLogsBtn.click();

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video: currentVideo.filename,
          mode: selectedMode,
          max_frames: selectedLength,
          conf: 0.12
        })
      });
      const data = await res.json();
      if (data.job_id) {
        activeJobId = data.job_id;
        pollInterval = setInterval(pollJobStatus, 800);
      } else {
        throw new Error(data.error || "Failed to start job");
      }
    } catch (err) {
      console.error(err);
      jobStageText.textContent = `Error: ${err.message}`;
      runPipelineBtn.disabled = false;
      runBtnText.textContent = "Run Pipeline";
    }
  });

  async function pollJobStatus() {
    if (!activeJobId) return;

    try {
      const res = await fetch(`/api/job/${activeJobId}`);
      const job = await res.json();

      // Update progress
      jobProgressBar.style.width = `${job.progress}%`;
      jobPercentText.textContent = `${job.progress}%`;

      if (job.fps && job.current_frame) {
        jobTelemetryText.textContent = `Frame ${job.current_frame}/${job.total_frames} | ${job.fps} | ${job.active_tracks || job.players || ''}`;
      }

      // Update terminal logs
      if (job.logs && job.logs.length > 0) {
        consoleLogs.textContent = job.logs.join("\n");
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
      }

      if (job.status === "completed") {
        clearInterval(pollInterval);
        activeJobId = null;
        jobStageText.textContent = "Complete!";
        jobProgressBar.style.width = "100%";
        jobPercentText.textContent = "100%";
        runPipelineBtn.disabled = false;
        runBtnText.textContent = "Run Pipeline";

        // Refresh data
        await loadVideos();
        loadHeatmaps();

        // Switch to heatmaps tab if tracking
        if (selectedMode === "track") {
          setTimeout(() => tabHeatmapsBtn.click(), 1000);
        }

        // Auto-play both videos
        setTimeout(() => {
          sourceVideo.currentTime = 0;
          if (outputVideo.src) {
            outputVideo.currentTime = 0;
            outputVideo.play();
          }
          sourceVideo.play();
          playIcon.classList.add("hidden");
          pauseIcon.classList.remove("hidden");
        }, 500);

      } else if (job.status === "error") {
        clearInterval(pollInterval);
        activeJobId = null;
        jobStageText.textContent = `Failed: ${job.error || 'Unknown error'}`;
        runPipelineBtn.disabled = false;
        runBtnText.textContent = "Run Pipeline";
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }

  // ── Heatmaps Drawer & Lightbox ────────────────────────────────────────────
  tabHeatmapsBtn.addEventListener("click", () => {
    tabHeatmapsBtn.classList.add("active");
    tabLogsBtn.classList.remove("active");
    heatmapsPanel.classList.add("active");
    logsPanel.classList.remove("active");
  });

  tabLogsBtn.addEventListener("click", () => {
    tabLogsBtn.classList.add("active");
    tabHeatmapsBtn.classList.remove("active");
    logsPanel.classList.add("active");
    heatmapsPanel.classList.remove("active");
  });

  async function loadHeatmaps() {
    try {
      const res = await fetch("/api/heatmaps");
      const data = await res.json();
      const heatmaps = data.heatmaps || [];

      if (heatmaps.length === 0) {
        heatmapsGrid.innerHTML = `<div class="empty-state">No heatmaps generated yet. Run tracking to produce occupancy heatmaps.</div>`;
        return;
      }

      heatmapsGrid.innerHTML = "";
      heatmaps.forEach((h) => {
        const card = document.createElement("div");
        card.className = "heatmap-card";
        card.innerHTML = `
          <img src="/heatmaps/${encodeURIComponent(h.filename)}?t=${Date.now()}" alt="${h.name}" class="heatmap-thumb" loading="lazy">
          <div class="heatmap-card-footer">
            <span class="heatmap-name">${h.name}</span>
            <span class="heatmap-size">${h.size_kb} KB</span>
          </div>
        `;
        card.addEventListener("click", () => {
          openModal(h.name, `/heatmaps/${encodeURIComponent(h.filename)}`);
        });
        heatmapsGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load heatmaps:", err);
    }
  }

  function openModal(title, imgSrc) {
    modalTitle.textContent = title;
    modalImage.src = imgSrc;
    imageModal.classList.remove("hidden");
  }

  function closeModal() {
    imageModal.classList.add("hidden");
    modalImage.src = "";
  }

  modalCloseBtn.addEventListener("click", closeModal);
  imageModal.addEventListener("click", (e) => {
    if (e.target === imageModal) closeModal();
  });

  // Init
  loadVideos();
});
