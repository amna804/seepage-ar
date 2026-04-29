/**
 * Wall Scanner AR App - Main Logic
 * ─────────────────────────────────
 * Flow:
 *  1. App opens → Load Models
 *  2. SCAN button clicked → Choose mode (pre-collected / real-time)
 *  3. Option 1: Upload CSV dataset (dry → dry test code, wet → wet test code)
 *  4. Run detection model → if wet, run localization model
 *  5. Return coordinates → mark quadrants on AR grid (red=wet)
 *  6. Send results to Unity via AR bridge
 */

// ─────────────────────────────────────────────
// CONFIG — update GITHUB_BASE to your repo URL
// ─────────────────────────────────────────────
const CONFIG = {
  GITHUB_BASE: "https://amna804.github.io/seepage-ar",
  DETECTION_MODEL_PATH:     "/models/detection/model.json",
  LOCALIZATION_MODEL_PATH:  "/models/localization/model.json",
  DETECTION_SCALER_PATH:    "/models/detection/scaler_params.json",
  LOCALIZATION_SCALER_PATH: "/models/localization/scaler_params.json",

  // AR Grid dimensions: X ∈ [-2,-1,0,1,2], Y ∈ [-1,0,1]
  GRID_X: [-2, -1, 0, 1, 2],
  GRID_Y: [-1, 0, 1],

  // Detection threshold
  WET_THRESHOLD: 0.5,
};

// ─────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────
const STATE = {
  detectionModel:     null,
  localizationModel:  null,
  detectionScaler:    null,
  localizationScaler: null,
  modelsReady:        false,
  scanning:           false,
  uploadedData:       null,
  wallType:           null,   // 'dry' | 'wet'
  results:            [],
};

// ─────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────
function log(msg, level = "info") {
  const console_ = document.getElementById("log-console");
  if (!console_) return;
  const time = new Date().toLocaleTimeString("en-US", { hour12: false });
  const line = document.createElement("div");
  line.className = `log-line ${level}`;
  line.innerHTML = `<span class="ts">[${time}]</span>${msg}`;
  console_.appendChild(line);
  console_.scrollTop = console_.scrollHeight;
}

function setStatus(msg, state = "ready") {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  if (dot)  { dot.className = "status-dot" + (state === "loading" ? " loading" : state === "error" ? " error" : ""); }
  if (text) text.textContent = msg;
}

function showSection(id) {
  document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
  const el = document.getElementById(id);
  if (el) el.classList.add("active");
  updateSteps(id);
}

function updateSteps(sectionId) {
  const order = ["section-home", "section-options", "section-upload", "section-scanning", "section-results"];
  const idx   = order.indexOf(sectionId);
  document.querySelectorAll(".step-dot").forEach((dot, i) => {
    dot.classList.toggle("done",   i < idx);
    dot.classList.toggle("active", i === idx);
  });
}

function setProgress(pct) {
  const bar = document.getElementById("progress-bar");
  if (bar) bar.style.width = pct + "%";
}

// ─────────────────────────────────────────────
// MODEL LOADING
// ─────────────────────────────────────────────
async function loadModels() {
  setStatus("Loading models...", "loading");
  log("Loading TF.js models from GitHub...", "info");

  try {
    const base = CONFIG.GITHUB_BASE;

    // Load scalers
    const [detScaler, locScaler] = await Promise.all([
      fetch(base + CONFIG.DETECTION_SCALER_PATH).then(r => r.json()),
      fetch(base + CONFIG.LOCALIZATION_SCALER_PATH).then(r => r.json()),
    ]);
    STATE.detectionScaler    = detScaler;
    STATE.localizationScaler = locScaler;
    log("✓ Scalers loaded", "success");

    // Load TF.js models
    STATE.detectionModel    = await tf.loadLayersModel(base + CONFIG.DETECTION_MODEL_PATH);
    log("✓ Detection model loaded", "success");

    STATE.localizationModel = await tf.loadLayersModel(base + CONFIG.LOCALIZATION_MODEL_PATH);
    log("✓ Localization model loaded", "success");

    STATE.modelsReady = true;
    setStatus("Models ready — Press SCAN to begin", "ready");
    document.getElementById("btn-scan").disabled = false;
    log("System ready!", "success");

  } catch (err) {
    setStatus("Model load failed", "error");
    log("ERROR: " + err.message, "error");
    log("Check GITHUB_BASE URL in app.js", "warn");
    console.error(err);
  }
}

// ─────────────────────────────────────────────
// SCALER — StandardScaler normalize
// ─────────────────────────────────────────────
function applyScaler(data, scaler) {
  return data.map((row) =>
    row.map((val, i) => (val - scaler.mean[i]) / (scaler.std[i] || 1))
  );
}

// ─────────────────────────────────────────────
// CSV PARSING
// ─────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split("\n").filter(l => l.trim());
  const header = lines[0].split(",").map(h => h.trim().toLowerCase());
  const rows = lines.slice(1).map(line => {
    const vals = line.split(",").map(v => parseFloat(v.trim()));
    return vals;
  }).filter(row => row.every(v => !isNaN(v)));
  log(`Parsed CSV: ${rows.length} samples, ${header.length} features`, "info");
  return { header, rows };
}

// ─────────────────────────────────────────────
// WALL TYPE DETECTION FROM FILENAME
// ─────────────────────────────────────────────
function detectWallTypeFromFilename(filename) {
  const name = filename.toLowerCase();
  if (name.includes("wet"))  return "wet";
  if (name.includes("dry"))  return "dry";
  return null;
}

// ─────────────────────────────────────────────
// DRY WALL TESTING PIPELINE
// ─────────────────────────────────────────────
async function runDryWallTest(data) {
  log("Running DRY wall test pipeline...", "info");
  const scaled = applyScaler(data.rows, STATE.detectionScaler);
  const tensor  = tf.tensor2d(scaled);
  const preds   = await STATE.detectionModel.predict(tensor).data();
  tensor.dispose();

  const wetCount = Array.from(preds).filter(p => p > CONFIG.WET_THRESHOLD).length;
  log(`Detection complete: ${wetCount}/${preds.length} samples above wet threshold`, "info");

  if (wetCount === 0) {
    log("RESULT: Wall is DRY — No moisture detected", "success");
    return { wallType: "dry", wetCoordinates: [] };
  } else {
    log(`WARN: ${wetCount} anomalous samples found in supposed dry wall`, "warn");
    return { wallType: "dry", wetCoordinates: [] }; // Still dry pipeline
  }
}

// ─────────────────────────────────────────────
// WET WALL TESTING PIPELINE
// ─────────────────────────────────────────────
async function runWetWallTest(data) {
  log("Running WET wall test pipeline...", "info");

  // Step 1: Detection
  const scaledDet   = applyScaler(data.rows, STATE.detectionScaler);
  const detTensor   = tf.tensor2d(scaledDet);
  const detPreds    = await STATE.detectionModel.predict(detTensor).data();
  detTensor.dispose();
  log("Detection model inference complete", "info");

  // Step 2: Localization for wet samples
  const wetCoordinates = [];
  const scaledLoc = applyScaler(data.rows, STATE.localizationScaler);

  for (let i = 0; i < scaledDet.length; i++) {
    setProgress(10 + (i / scaledDet.length) * 70);
    const isWet = detPreds[i] > CONFIG.WET_THRESHOLD;
    if (!isWet) continue;

    const locTensor = tf.tensor2d([scaledLoc[i]]);
    const locPred   = await STATE.localizationModel.predict(locTensor).data();
    locTensor.dispose();

    // Model outputs [x, y] coordinates
    const rawX = locPred[0], rawY = locPred[1];

    // Snap to nearest grid point
    const x = snapToGrid(rawX, CONFIG.GRID_X);
    const y = snapToGrid(rawY, CONFIG.GRID_Y);

    // Avoid duplicates
    if (!wetCoordinates.some(c => c.x === x && c.y === y)) {
      wetCoordinates.push({ x, y, confidence: detPreds[i] });
      log(`Wet spot detected at (x=${x}, y=${y}) conf=${(detPreds[i]*100).toFixed(1)}%`, "warn");
    }
  }

  log(`RESULT: ${wetCoordinates.length} wet zone(s) found`, wetCoordinates.length > 0 ? "warn" : "success");
  return { wallType: "wet", wetCoordinates };
}

function snapToGrid(value, gridValues) {
  return gridValues.reduce((prev, curr) =>
    Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev
  );
}

// ─────────────────────────────────────────────
// MAIN SCAN ORCHESTRATOR
// ─────────────────────────────────────────────
async function runScan(file) {
  if (!STATE.modelsReady) { log("Models not ready!", "error"); return; }

  showSection("section-scanning");
  setProgress(5);
  STATE.scanning = true;

  try {
    const text = await file.text();
    const data = parseCSV(text);
    setProgress(10);

    const wallType = STATE.wallType || detectWallTypeFromFilename(file.name);
    if (!wallType) {
      log("Cannot determine wall type from filename. Name file with 'dry' or 'wet'.", "error");
      setStatus("Filename must contain 'dry' or 'wet'", "error");
      return;
    }

    log(`Wall type from filename: ${wallType.toUpperCase()}`, "info");
    STATE.wallType = wallType;

    let result;
    if (wallType === "dry") {
      result = await runDryWallTest(data);
    } else {
      result = await runWetWallTest(data);
    }

    setProgress(100);
    STATE.results = result;
    setTimeout(() => showResults(result), 400);

    // Unity AR Bridge
    sendToUnity(result);

  } catch (err) {
    log("Scan error: " + err.message, "error");
    setStatus("Scan failed", "error");
    console.error(err);
  } finally {
    STATE.scanning = false;
  }
}

// ─────────────────────────────────────────────
// RESULTS DISPLAY
// ─────────────────────────────────────────────
function showResults(result) {
  showSection("section-results");
  renderARGrid(result);
  renderResultSummary(result);
}

function renderARGrid(result) {
  const grid = document.getElementById("ar-grid");
  if (!grid) return;

  // Clear old points
  grid.querySelectorAll(".quadrant-point").forEach(el => el.remove());

  const { GRID_X, GRID_Y } = CONFIG;
  const cols = GRID_X.length;
  const rows = GRID_Y.length;

  GRID_Y.slice().reverse().forEach((y, rowIdx) => {
    GRID_X.forEach((x, colIdx) => {
      const isWet = result.wetCoordinates.some(c => c.x === x && c.y === y);
      const pct_x = (colIdx / (cols - 1)) * 100;
      const pct_y = (rowIdx / (rows - 1)) * 100;

      const point = document.createElement("div");
      point.className = `quadrant-point ${isWet ? "wet" : "dry"}`;
      point.style.left = `${pct_x}%`;
      point.style.top  = `${pct_y}%`;
      point.title = `(${x}, ${y})${isWet ? " — WET" : ""}`;
      point.textContent = isWet ? "💧" : "";
      grid.appendChild(point);
    });
  });

  // Add axis labels
  GRID_X.forEach((x, i) => {
    const lbl = document.createElement("div");
    lbl.className = "grid-label-x";
    lbl.style.left   = `${(i / (GRID_X.length - 1)) * 100}%`;
    lbl.style.bottom = "2px";
    lbl.style.transform = "translateX(-50%)";
    lbl.textContent = x;
    grid.appendChild(lbl);
  });

  GRID_Y.slice().reverse().forEach((y, i) => {
    const lbl = document.createElement("div");
    lbl.className = "grid-label-y";
    lbl.style.top   = `${(i / (GRID_Y.length - 1)) * 100}%`;
    lbl.style.left  = "2px";
    lbl.style.transform = "translateY(-50%)";
    lbl.textContent = y;
    grid.appendChild(lbl);
  });
}

function renderResultSummary(result) {
  const el = document.getElementById("result-summary");
  if (!el) return;

  const isWet = result.wallType === "wet" && result.wetCoordinates.length > 0;

  if (!isWet) {
    el.className = "result-summary dry";
    el.innerHTML = `
      <div class="result-icon">✅</div>
      <div class="result-title" style="color: var(--success)">Wall is DRY — Secure</div>
      <div class="result-desc">No moisture detected in the scanned wall.</div>
    `;
  } else {
    const badges = result.wetCoordinates.map(c =>
      `<span class="coord-badge">(${c.x}, ${c.y})</span>`
    ).join("");
    el.className = "result-summary wet";
    el.innerHTML = `
      <div class="result-icon">💧</div>
      <div class="result-title" style="color: var(--danger)">Moisture Detected!</div>
      <div class="result-desc">Wet zones found at coordinates:</div>
      <div class="wet-coords">${badges}</div>
    `;
  }
}

// ─────────────────────────────────────────────
// UNITY AR BRIDGE
// ─────────────────────────────────────────────
function sendToUnity(result) {
  const payload = JSON.stringify(result);

  // Method 1: vuplex (common Unity WebView bridge)
  if (window.vuplex) {
    window.vuplex.postMessage(payload);
    log("Results sent to Unity via vuplex", "success");
    return;
  }

  // Method 2: Unity SendMessage (if using Unity WebView plugin)
  if (typeof window.Unity !== "undefined") {
    window.Unity.call(payload);
    log("Results sent to Unity via Unity.call", "success");
    return;
  }

  // Method 3: Custom event (pick up in Unity C# via JavaScript bridge)
  window.dispatchEvent(new CustomEvent("wallScanResult", { detail: result }));
  log("Results dispatched via CustomEvent 'wallScanResult'", "info");
  log("Unity should listen: window.addEventListener('wallScanResult', ...)", "info");
}

// ─────────────────────────────────────────────
// UI EVENT HANDLERS
// ─────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {

  // Load models on start
  loadModels();

  // SCAN button
  document.getElementById("btn-scan")?.addEventListener("click", () => {
    showSection("section-options");
    log("Scan options displayed", "info");
  });

  // Option 1: Pre-collected dataset
  document.getElementById("btn-option-precollected")?.addEventListener("click", () => {
    showSection("section-upload");
    log("Mode: Pre-collected dataset", "info");
  });

  // Option 2: Real-time (placeholder)
  document.getElementById("btn-option-realtime")?.addEventListener("click", () => {
    log("Real-time collection not available in this build", "warn");
    alert("Real-time data collection requires hardware connection.\nThis build supports pre-collected datasets only.");
  });

  // Back buttons
  document.querySelectorAll(".btn-back").forEach(btn => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.target || "section-home";
      showSection(target);
    });
  });

  // File upload
  const fileInput = document.getElementById("file-input");
  const uploadZone = document.getElementById("upload-zone");

  fileInput?.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) handleFileSelected(file);
  });

  uploadZone?.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });

  uploadZone?.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
  });

  uploadZone?.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelected(file);
  });

  // Start scan button
  document.getElementById("btn-start-scan")?.addEventListener("click", () => {
    if (STATE.uploadedData) runScan(STATE.uploadedData);
  });

  // New scan
  document.getElementById("btn-new-scan")?.addEventListener("click", () => {
    STATE.uploadedData = null;
    STATE.wallType = null;
    STATE.results = [];
    setProgress(0);
    showSection("section-home");
    log("--- New scan session ---", "info");
  });
});

function handleFileSelected(file) {
  STATE.uploadedData = file;
  const nameEl = document.getElementById("upload-filename");
  if (nameEl) nameEl.textContent = file.name;

  const wallType = detectWallTypeFromFilename(file.name);
  STATE.wallType = wallType;

  if (wallType) {
    log(`File "${file.name}" detected as: ${wallType.toUpperCase()} wall`, "success");
  } else {
    log(`WARN: Cannot detect wall type from "${file.name}". Rename to include 'dry' or 'wet'.`, "warn");
  }

  document.getElementById("btn-start-scan").disabled = false;
}

// Expose for Unity to poll if needed
window.WallScanner = {
  getResults: () => STATE.results,
  isReady:    () => STATE.modelsReady,
};
