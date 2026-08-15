import { Niivue, SLICE_TYPE } from "@niivue/niivue";
import DOMPurify from "dompurify";
import { marked } from "marked";
import "./style.css";

const $ = (selector) => document.querySelector(selector);
const form = $("#analysis-form");
const t1Input = $("#t1-input");
const fmriInput = $("#fmri-input");
const analyzeButton = $("#analyze-button");
const analyzeButtonTooltip = analyzeButton.dataset.tooltip;
const thresholdInput = $("#threshold");
const dynamicSensitivityInput = $("#dynamic-sensitivity");
const opacityInput = $("#opacity");
const backgroundSelect = $("#background-select");
const runNameInput = $("#run-name");
const toast = $("#toast");
const appTooltip = $("#app-tooltip");
let activeJob = null;
let activeJobWasReused = false;
let pollTimer = null;
let resultLoaded = false;
let selectedPair = null;
let currentResultJob = null;
let pairRecords = [];
let resultRecords = [];
let llmConfigured = false;
let viewerMode = "slices";
let mapMode = "dynamic";
let brainPalette = "warm";
const DEFAULT_BRAIN_PALETTES = Object.freeze({
  dynamic: "warm",
  activity: "cool",
});
let glassPalette = "viridis";
let currentResult = null;
let viewerAvailable = true;
let glassViewerAttached = false;
let glassViewerUnavailable = false;
let glassLoadedJob = null;
let glassLoadPromise = null;
let currentFunctionalPrediction = null;
let predictionFocusedRegionId = null;
let predictionViewerAttached = false;
let predictionViewerUnavailable = false;
let predictionLoadedId = null;
let predictionViewMode = "slices";
let predictionBrainPalette = "warm";
let predictionGlassPalette = "viridis";
let predictionGlassRendererAttached = false;
let predictionGlassRendererUnavailable = false;
let predictionGlassLoadedId = null;
let predictionGlassLoadPromise = null;
let predictionGlassMipRequest = null;
let predictionGlassPreloadHandle = null;
let predictionGlassPreloadUsesIdleCallback = false;
let functionalPredictionRecords = [];
let predictionProgressInterval = null;
let predictionProgressStartedAt = 0;
let predictionProgressGeneratingStartedAt = 0;
let predictionProgressStageId = "preparing";
let predictionProgressPercent = 0;
let predictionProgressCompletedStages = new Set();
let predictionUsagePhase = "waiting";
let functionalContrastCatalogLoaded = false;
let boldRequestTimer = null;
let boldRequestController = null;
let boldPlotState = null;
let boldResizeFrame = null;
let appTooltipTarget = null;
let nextPairName = "Pair 1";
const boldSeriesCache = new Map();
const SVG_NS = "http://www.w3.org/2000/svg";
const BOLD_PLOT = {
  width: 420,
  height: 190,
  left: 64,
  right: 14,
  top: 10,
  bottom: 34,
};

const SENSITIVITY_THRESHOLDS = Object.freeze({
  "-3": 99,
  "-2": 97,
  "-1": 94,
  0: 90,
  1: 85,
  2: 80,
  3: 70,
});
const calibratedThreshold = (control) => (
  SENSITIVITY_THRESHOLDS[control.value] ?? SENSITIVITY_THRESHOLDS[0]
);
const dynamicThreshold = () => calibratedThreshold(dynamicSensitivityInput);
const activityThreshold = () => calibratedThreshold(thresholdInput);
const selectedThreshold = () => mapMode === "dynamic"
  ? dynamicThreshold()
  : activityThreshold();
const selectedOverlayOpacity = () => Number(opacityInput.value) / 100;
// Keep NiiVue's native 1× scale; the responsive canvas supplies full-size rendering
// without stretching a previously rendered bitmap.
const INITIAL_3D_SCALE = 1;
const PREDICTION_THRESHOLDS = Object.freeze({
  "-3": 1.64,
  "-2": 1.28,
  "-1": 1.04,
  0: 0.67,
  1: 0.39,
  2: 0.13,
  3: 0.03,
});
const predictionScoreMaximum = () => {
  const maximum = Number(currentFunctionalPrediction?.score_scale?.maximum);
  return Number.isFinite(maximum) && maximum > 0 ? maximum : 2.58;
};
const predictionThreshold = () => {
  const thresholds = currentFunctionalPrediction?.score_scale?.sensitivity_thresholds
    || PREDICTION_THRESHOLDS;
  const threshold = Number(
    thresholds[$("#prediction-sensitivity").value]
    ?? thresholds[0]
    ?? PREDICTION_THRESHOLDS[0],
  );
  return Number.isFinite(threshold) ? threshold : PREDICTION_THRESHOLDS[0];
};
const predictionOverlayOpacity = () => Number($("#prediction-opacity").value) / 100;
const PREDICTION_PROGRESS_STEPS = Object.freeze([
  { id: "preparing", label: "Preparing the NeuroTaskFM workspace", start: 0, end: 5 },
  { id: "target", label: "Loading the target group reference", start: 5, end: 15 },
  { id: "source", label: "Reading the source pattern", start: 15, end: 30 },
  { id: "generating", label: "Generating the functional prediction", start: 30, end: 90 },
  { id: "visualizing", label: "Visualizing the final output", start: 90, end: 100 },
]);

async function loadFunctionalContrastCatalog() {
  const select = $("#prediction-target-task");
  const status = $("#prediction-contrast-status");
  select.disabled = true;
  try {
    const response = await fetch("/api/functional-contrasts");
    const payload = await response.json();
    if (!response.ok || !Array.isArray(payload.contrasts)) {
      throw new Error(payload.message || "Contrast catalog is unavailable");
    }
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose a target contrast";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);

    const groups = new Map();
    payload.contrasts.forEach((contrast) => {
      const groupLabel = contrast.dataset === "hcp"
        ? `HCP · ${contrast.task}`
        : `${contrast.dataset_label} · ${contrast.task_label}`;
      let group = groups.get(groupLabel);
      if (!group) {
        group = document.createElement("optgroup");
        group.label = groupLabel;
        groups.set(groupLabel, group);
        select.appendChild(group);
      }
      const option = document.createElement("option");
      option.value = contrast.id;
      option.textContent = contrast.contrast;
      option.dataset.description = contrast.description || "";
      option.dataset.displayName = contrast.display_name || contrast.contrast;
      group.appendChild(option);
    });
    functionalContrastCatalogLoaded = true;
    const selectedContrast = currentFunctionalPrediction?.target_contrast_id;
    if (selectedContrast && select.querySelector(`option[value="${CSS.escape(selectedContrast)}"]`)) {
      select.value = selectedContrast;
    }
    status.textContent = "";
    status.classList.add("hidden");
  } catch (error) {
    select.replaceChildren();
    const unavailable = document.createElement("option");
    unavailable.value = "";
    unavailable.textContent = "Contrast catalog unavailable";
    unavailable.disabled = true;
    unavailable.selected = true;
    select.appendChild(unavailable);
    status.textContent = error.message;
    status.classList.remove("hidden");
    functionalContrastCatalogLoaded = false;
  } finally {
    select.disabled = !functionalContrastCatalogLoaded;
  }
}

function renderFunctionalPredictionHistory() {
  const list = $("#prediction-history-list");
  const count = $("#prediction-history-count");
  count.textContent = `${functionalPredictionRecords.length}/10 saved`;
  if (!functionalPredictionRecords.length) {
    list.innerHTML = '<div class="prediction-history-empty">No saved predictions yet.</div>';
    return;
  }
  list.replaceChildren(...functionalPredictionRecords.map((prediction) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "prediction-history-item";
    const selected = prediction.prediction_id === currentFunctionalPrediction?.prediction_id;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
    button.dataset.tooltip = `Load Saved Prediction: ${prediction.target_label || prediction.target_contrast || "Contrast"}`;

    const title = document.createElement("span");
    title.className = "prediction-history-title";
    title.textContent = prediction.target_label || prediction.target_contrast || "Saved prediction";
    const meta = document.createElement("span");
    meta.className = "prediction-history-meta";
    meta.textContent = [
      formatDate(prediction.created_at),
      prediction.source_task,
    ].filter(Boolean).join(" · ");
    const action = document.createElement("span");
    action.className = "prediction-history-action";
    action.textContent = selected ? "Open" : "Load";
    button.append(title, meta, action);
    button.addEventListener("click", () => loadSavedFunctionalPrediction(prediction.prediction_id));
    return button;
  }));
}

async function loadFunctionalPredictionHistory() {
  const jobId = currentResultJob;
  if (!jobId || !resultLoaded) {
    functionalPredictionRecords = [];
    renderFunctionalPredictionHistory();
    return;
  }
  const list = $("#prediction-history-list");
  list.innerHTML = '<div class="prediction-history-empty">Loading saved predictions…</div>';
  try {
    const response = await fetch(`/api/jobs/${jobId}/functional-predictions`);
    const payload = await response.json();
    if (!response.ok || !Array.isArray(payload.predictions)) {
      throw new Error(payload.message || "Saved predictions are unavailable");
    }
    if (jobId !== currentResultJob) return;
    functionalPredictionRecords = payload.predictions.slice(0, 10);
    renderFunctionalPredictionHistory();
  } catch (error) {
    if (jobId !== currentResultJob) return;
    functionalPredictionRecords = [];
    renderFunctionalPredictionHistory();
    showToast(error.message, true);
  }
}

async function loadSavedFunctionalPrediction(predictionId) {
  const jobId = currentResultJob;
  if (!jobId || !predictionId) return;
  const list = $("#prediction-history-list");
  list.setAttribute("aria-busy", "true");
  try {
    const response = await fetch(
      `/api/jobs/${jobId}/functional-predictions/${encodeURIComponent(predictionId)}`,
    );
    const prediction = await response.json();
    if (!response.ok) throw new Error(prediction.message || "Saved prediction is unavailable");
    if (jobId !== currentResultJob) return;
    if (!functionalContrastCatalogLoaded) await loadFunctionalContrastCatalog();
    $("#prediction-source-task").value = prediction.source_task || "";
    const contrast = $("#prediction-target-task");
    if (prediction.target_contrast_id
      && contrast.querySelector(`option[value="${CSS.escape(prediction.target_contrast_id)}"]`)) {
      contrast.value = prediction.target_contrast_id;
    }
    await renderFunctionalPrediction(prediction);
    showToast(`Loaded ${prediction.target_label || prediction.target_contrast || "saved prediction"}.`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    list.removeAttribute("aria-busy");
  }
}

function setRunName(value, source = "auto") {
  runNameInput.value = value || nextPairName;
  runNameInput.dataset.source = source;
}

function currentRunName() {
  return runNameInput.value.trim();
}

function positionAppTooltip(target) {
  const targetBounds = target.getBoundingClientRect();
  const tooltipBounds = appTooltip.getBoundingClientRect();
  const viewportPadding = 12;
  const gap = 10;
  const centeredLeft = targetBounds.left + targetBounds.width / 2 - tooltipBounds.width / 2;
  const left = Math.min(
    window.innerWidth - tooltipBounds.width - viewportPadding,
    Math.max(viewportPadding, centeredLeft),
  );
  let top = targetBounds.top - tooltipBounds.height - gap;
  if (top < viewportPadding) top = targetBounds.bottom + gap;
  top = Math.min(
    window.innerHeight - tooltipBounds.height - viewportPadding,
    Math.max(viewportPadding, top),
  );
  appTooltip.style.left = `${Math.round(left)}px`;
  appTooltip.style.top = `${Math.round(top)}px`;
}

function showAppTooltip(target) {
  const message = target?.dataset?.tooltip?.trim();
  if (!message) return;
  appTooltipTarget = target;
  const host = target.closest("dialog") || document.body;
  if (appTooltip.parentElement !== host) host.appendChild(appTooltip);
  appTooltip.classList.remove("visible");
  appTooltip.textContent = message;
  appTooltip.setAttribute("aria-hidden", "false");
  positionAppTooltip(target);
  appTooltip.classList.add("visible");
}

function hideAppTooltip() {
  appTooltipTarget = null;
  appTooltip.classList.remove("visible");
  appTooltip.setAttribute("aria-hidden", "true");
}

function tooltipTarget(event) {
  if (!(event.target instanceof Element)) return null;
  return event.target.closest("[data-tooltip]");
}

const viewerOptions = {
  backColor: [0.025, 0.035, 0.045, 1],
  fontColor: [0.91, 0.97, 0.94, 1],
  crosshairColor: [0.42, 0.93, 0.76, 0.9],
  selectionBoxColor: [0.42, 0.93, 0.76, 0.25],
  show3Dcrosshair: true,
  isRadiologicalConvention: false,
  loadingText: "Loading neuroimaging data",
};

const mniDisplayVolume = (name = "MNI152_1mm.nii.gz") => ({
  url: "/api/reference/mni152_1mm.nii.gz",
  name,
  colormap: "gray",
  opacity: 1,
  cal_min: 0.22,
  cal_max: 0.91,
  trustCalMinMax: true,
});

const slices = new Niivue({ ...viewerOptions, multiplanarForceRender: false });
const glassRenderer = new Niivue({
  ...viewerOptions,
  backColor: [0.01, 0.018, 0.015, 1],
  show3Dcrosshair: false,
  isColorbar: false,
  isOrientCube: false,
  dragAndDropEnabled: false,
  renderOverlayBlend: 1,
});
const predictionViewer = new Niivue({
  ...viewerOptions,
  show3Dcrosshair: true,
  multiplanarForceRender: false,
  dragAndDropEnabled: false,
});
const predictionGlassRenderer = new Niivue({
  ...viewerOptions,
  backColor: [0.01, 0.018, 0.015, 1],
  show3Dcrosshair: false,
  isColorbar: false,
  isOrientCube: false,
  dragAndDropEnabled: false,
  renderOverlayBlend: 1,
});
predictionViewer.onLocationChange = (location) => {
  if (!location?.mm) return;
  const roundedMm = location.mm.slice(0, 3).map((value) => Math.round(Number(value)));
  $("#prediction-coordinate-readout").textContent = `MNI   ${roundedMm.join(", ")}`;
};

const GLASS_3D_PALETTES = ["warm", "plasma", "viridis", "inferno", "turbo"];
const glass3DPaletteName = (palette) => `neurotaskfm-glass-${palette}-strong`;
for (const renderer of [glassRenderer, predictionGlassRenderer]) {
  GLASS_3D_PALETTES.forEach((palette) => {
    const source = renderer.colormapFromKey(palette);
    const lastIndex = Math.max(1, source.A.length - 1);
    renderer.addColormap(glass3DPaletteName(palette), {
      ...source,
      R: [...source.R],
      G: [...source.G],
      B: [...source.B],
      I: [...source.I],
      A: source.A.map((_, index) => (
        index === 0 ? 0 : Math.round(190 + (65 * (index - 1) / Math.max(1, lastIndex - 1)))
      )),
    });
  });
}

async function initializeViewers() {
  try {
    await slices.attachToCanvas($("#slice-canvas"));
    slices.setSliceType(SLICE_TYPE.MULTIPLANAR);
    slices.onLocationChange = (location) => {
      if (!location?.mm) return;
      const exactMm = location.mm.slice(0, 3).map(Number);
      const roundedMm = exactMm.map((value) => Math.round(value));
      $("#coordinate-readout").textContent = `MNI   ${roundedMm.join(", ")}`;
      if (resultLoaded && currentResultJob) {
        scheduleBoldTimecourse(exactMm);
      }
    };
    await loadReference();
    $("#slice-loader").classList.add("hidden");
  } catch (error) {
    viewerAvailable = false;
    $("#slice-loader span").textContent = "Brain Viewer requires WebGL2";
    $("#slice-loader i").classList.add("hidden");
    $("#glass-loader").classList.add("hidden");
    resetGlassBrain();
    resetAssistant();
    resetFunctionalPrediction(true);
    resetBoldPlot();
    showToast(`Brain Viewer unavailable: ${error.message}`, true);
  }
  await refreshLibrary();
  const resultId = new URLSearchParams(window.location.hash.slice(1)).get("result");
  if (resultId) await openSavedResult(resultId, false);
}

async function loadSliceVolumes(volumeFactory) {
  await slices.loadVolumes(volumeFactory());
  slices.setScale(INITIAL_3D_SCALE);
}

const mniRenderVolume = () => ({
  url: "/api/reference/mni152_2mm.nii.gz",
  name: "MNI152_2mm_transparent.nii.gz",
  colormap: "gray",
  opacity: 0.24,
  cal_min: 0.38,
  cal_max: 0.90,
  trustCalMinMax: true,
});

function glassResultVolumes(jobId, result = {}) {
  const hasRegionalDisplay = Boolean(result.outputs?.display_loading_mni);
  const displayMap = hasRegionalDisplay ? "display_loading_mni.nii.gz" : "loading_mni.nii.gz";
  const volumes = [
    mniRenderVolume(),
    {
      url: `/api/jobs/${jobId}/files/outputs/${displayMap}`,
      name: "Dynamic_Task_ROI_3D.nii.gz",
      colormap: glass3DPaletteName(glassPalette),
      opacity: mapMode === "dynamic" ? selectedOverlayOpacity() : 0,
      cal_min: dynamicThreshold(),
      cal_max: 100,
      trustCalMinMax: true,
    },
  ];
  if (result.outputs?.activity_mni) {
    volumes.push({
      url: `/api/jobs/${jobId}/files/outputs/activity_mni.nii.gz`,
      name: "All_Active_Voxels_3D.nii.gz",
      colormap: glass3DPaletteName(glassPalette),
      opacity: mapMode === "activity" ? selectedOverlayOpacity() : 0,
      cal_min: activityThreshold(),
      cal_max: 100,
      trustCalMinMax: true,
    });
  }
  return volumes;
}

function updateGlassDisplayControls() {
  if (!glassViewerAttached || glassLoadedJob !== currentResultJob) return;
  if (glassRenderer.volumes.length < 2) return;
  const overlayOpacity = selectedOverlayOpacity();
  const strongPalette = glass3DPaletteName(glassPalette);
  glassRenderer.setOpacity(0, 0.24);
  glassRenderer.volumes[1].cal_min = dynamicThreshold();
  glassRenderer.volumes[1].cal_max = 100;
  if (glassRenderer.volumes[1].colormap !== strongPalette) {
    glassRenderer.setColormap(glassRenderer.volumes[1].id, strongPalette);
  }
  glassRenderer.setOpacity(1, mapMode === "dynamic" ? overlayOpacity : 0);
  if (glassRenderer.volumes[2]) {
    glassRenderer.volumes[2].cal_min = activityThreshold();
    glassRenderer.volumes[2].cal_max = 100;
    if (glassRenderer.volumes[2].colormap !== strongPalette) {
      glassRenderer.setColormap(glassRenderer.volumes[2].id, strongPalette);
    }
    glassRenderer.setOpacity(2, mapMode === "activity" ? overlayOpacity : 0);
  }
  glassRenderer.updateGLVolume();
  glassRenderer.drawScene();
}

async function loadGlassRenderer(jobId, result = {}) {
  const loader = $("#glass-render-loader");
  const fallback = $("#glass-render-fallback");
  if (!viewerAvailable || glassViewerUnavailable) {
    loader.classList.add("hidden");
    fallback.classList.remove("hidden");
    return;
  }
  if (glassLoadPromise) {
    try {
      await glassLoadPromise;
    } catch {
      loader.classList.add("hidden");
      fallback.classList.remove("hidden");
      return;
    }
  }
  if (glassLoadedJob === jobId) {
    fallback.classList.add("hidden");
    updateGlassDisplayControls();
    window.requestAnimationFrame(() => {
      glassRenderer.resizeListener();
      glassRenderer.drawScene();
    });
    return;
  }

  loader.classList.remove("hidden");
  fallback.classList.add("hidden");
  glassLoadPromise = (async () => {
    if (!glassViewerAttached) {
      await glassRenderer.attachToCanvas($("#glass-render-canvas"));
      glassViewerAttached = true;
    }
    await glassRenderer.loadVolumes(glassResultVolumes(jobId, result));
    glassLoadedJob = jobId;
    glassRenderer.setSliceType(SLICE_TYPE.RENDER);
    glassRenderer.setRenderAzimuthElevation(-55, 22);
    glassRenderer.setScale(INITIAL_3D_SCALE);
    updateGlassDisplayControls();
  })();
  try {
    await glassLoadPromise;
  } catch (error) {
    glassViewerUnavailable = true;
    fallback.classList.remove("hidden");
    showToast(`Interactive 3D unavailable: ${error.message}`, true);
  } finally {
    glassLoadPromise = null;
    loader.classList.add("hidden");
  }
}

function resetGlassBrain(message = "Run or open an analysis to create the all-depth glass-brain views") {
  const image = $("#glass-brain-image");
  $("#glass-layout").classList.add("hidden");
  image.classList.remove("visible");
  image.removeAttribute("src");
  delete image.dataset.src;
  delete image.dataset.pendingSrc;
  $("#glass-placeholder").classList.remove("hidden");
  $("#glass-placeholder small").textContent = message;
  $("#glass-loader").classList.add("hidden");
  $("#glass-mip-fallback").classList.add("hidden");
  $("#glass-render-loader").classList.add("hidden");
  $("#glass-render-fallback").classList.add("hidden");
}

function resetAssistant(message = "Open an analyzed result to ask about its task, networks, and loading pattern.") {
  const list = $("#chat-list");
  list.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "chat-empty";
  const glyph = document.createElement("span");
  glyph.textContent = "✦";
  const copy = document.createElement("p");
  copy.textContent = message;
  empty.append(glyph, copy);
  list.appendChild(empty);
  for (const selector of ["#assistant-question", "#assistant-submit"]) {
    $(selector).disabled = true;
  }
}

function updateFunctionalPredictionAvailability() {
  $("#open-functional-prediction").disabled = !(
    resultLoaded && currentResultJob && llmConfigured
  );
}

function formatCompactCount(value) {
  const count = Math.max(0, Math.round(Number(value) || 0));
  return count > 1000 ? `${(count / 1000).toFixed(1)}k` : String(count);
}

function predictionProgressStep(stageId = predictionProgressStageId) {
  return PREDICTION_PROGRESS_STEPS.find((step) => step.id === stageId)
    || PREDICTION_PROGRESS_STEPS[0];
}

function setPredictionProgressPercent(value) {
  const percentage = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  predictionProgressPercent = Math.max(predictionProgressPercent, percentage);
  const detail = predictionProgressStep().label;
  const track = $("#prediction-progress-track");
  $("#prediction-progress-step").textContent = `${predictionProgressPercent}%`;
  $("#prediction-progress-fill").style.width = `${predictionProgressPercent}%`;
  track.setAttribute("aria-valuenow", String(predictionProgressPercent));
  track.setAttribute(
    "aria-valuetext",
    `${predictionProgressPercent} percent: ${detail}`,
  );
}

function renderPredictionProgressStages() {
  const activeIndex = PREDICTION_PROGRESS_STEPS.findIndex(
    (step) => step.id === predictionProgressStageId,
  );
  document.querySelectorAll("[data-prediction-stage]").forEach((item) => {
    const index = PREDICTION_PROGRESS_STEPS.findIndex(
      (step) => step.id === item.dataset.predictionStage,
    );
    const completed = predictionProgressCompletedStages.has(item.dataset.predictionStage)
      || (activeIndex >= 0 && index < activeIndex);
    item.classList.toggle("is-complete", completed);
    item.classList.toggle(
      "is-active",
      item.dataset.predictionStage === predictionProgressStageId && !completed,
    );
  });
}

function setPredictionProgressStage(stageId, state = "active", percent = null) {
  const step = PREDICTION_PROGRESS_STEPS.find((candidate) => candidate.id === stageId);
  if (!step) return;
  predictionProgressStageId = step.id;
  if (state === "complete") predictionProgressCompletedStages.add(step.id);
  if (state === "active" && step.id === "generating" && !predictionProgressGeneratingStartedAt) {
    predictionProgressGeneratingStartedAt = Date.now();
    setPredictionUsagePhase("thinking");
  }
  if (state === "active" && step.id === "visualizing") setPredictionUsagePhase("complete");
  $("#prediction-progress-detail").textContent = step.label;
  renderPredictionProgressStages();
  setPredictionProgressPercent(percent ?? (state === "complete" ? step.end : step.start));
}

function setPredictionUsage(kind, value, exact = false) {
  const count = Number(value);
  if (!Number.isFinite(count) || count < 0) return;
  const container = $(`#prediction-usage-${kind}`);
  if (!container) return;
  const rounded = Math.round(count);
  container.dataset.exact = String(Boolean(exact));
  container.classList.toggle("is-complete", Boolean(exact));
  const label = {
    input: "Input",
    reasoning: "Thinking",
    output: "Output",
  }[kind] || kind;
  container.setAttribute(
    "aria-label",
    `${exact ? "Exact" : "Live estimated"} ${label.toLowerCase()} token count ${rounded}`,
  );
  $(`#prediction-usage-${kind}-count`).textContent = formatCompactCount(rounded);
}

function setPredictionUsagePhase(phase) {
  if (!["waiting", "thinking", "output", "complete"].includes(phase)) return;
  predictionUsagePhase = phase;
  const activeKind = phase === "thinking" ? "reasoning" : phase === "output" ? "output" : null;
  for (const kind of ["input", "reasoning", "output"]) {
    const container = $(`#prediction-usage-${kind}`);
    container.classList.toggle("is-active", kind === activeKind);
  }
  const reasoning = $("#prediction-usage-reasoning");
  const reasoningCount = $("#prediction-usage-reasoning-count");
  if (phase === "thinking" && reasoning.dataset.exact !== "true") {
    reasoningCount.textContent = "…";
    reasoning.setAttribute("aria-label", "Thinking in progress; exact token count arrives on completion");
  } else if (phase === "output" && reasoning.dataset.exact !== "true") {
    reasoningCount.textContent = "Done";
    reasoning.setAttribute("aria-label", "Thinking complete; exact token count pending");
  }
  if (phase === "output" && $("#prediction-usage-output-count").textContent === "—") {
    $("#prediction-usage-output-count").textContent = "0";
  }
}

function resetPredictionUsage() {
  for (const kind of ["input", "reasoning", "output"]) {
    const container = $(`#prediction-usage-${kind}`);
    container.removeAttribute("data-exact");
    container.classList.remove("is-active", "is-complete");
    $(`#prediction-usage-${kind}-count`).textContent = "—";
  }
  setPredictionUsagePhase("waiting");
}

function renderPredictionResultUsage(usage = {}) {
  const container = $("#prediction-result-usage");
  const values = {
    input: Number(usage.input_count),
    reasoning: Number(usage.reasoning_count),
    output: Number(usage.output_count),
  };
  const hasUsage = Object.values(values).some((value) => Number.isFinite(value) && value >= 0);
  container.classList.toggle("hidden", !hasUsage);
  if (!hasUsage) return;
  for (const [kind, value] of Object.entries(values)) {
    $(`#prediction-result-${kind}-count`).textContent = (
      Number.isFinite(value) && value >= 0 ? formatCompactCount(value) : "—"
    );
  }
  container.setAttribute(
    "aria-label",
    [
      Number.isFinite(values.input) ? `${Math.round(values.input)} input tokens` : "",
      Number.isFinite(values.reasoning) ? `${Math.round(values.reasoning)} thinking tokens` : "",
      Number.isFinite(values.output) ? `${Math.round(values.output)} output tokens` : "",
    ].filter(Boolean).join(", "),
  );
}

function updatePredictionProgressClock() {
  const elapsedSeconds = Math.floor((Date.now() - predictionProgressStartedAt) / 1000);
  $("#prediction-progress-elapsed").textContent = `Elapsed ${formatRuntime(elapsedSeconds)}`;
  if (
    predictionProgressStageId === "generating"
    && !predictionProgressCompletedStages.has("generating")
    && predictionProgressGeneratingStartedAt
  ) {
    const generatingElapsed = Date.now() - predictionProgressGeneratingStartedAt;
    const estimatedPercentage = Math.min(89, 30 + Math.floor(generatingElapsed / 1500));
    setPredictionProgressPercent(estimatedPercentage);
  }
}

function stopPredictionProgress(reset = false) {
  if (predictionProgressInterval !== null) {
    window.clearInterval(predictionProgressInterval);
    predictionProgressInterval = null;
  }
  if (reset) {
    predictionProgressStartedAt = Date.now();
    predictionProgressGeneratingStartedAt = 0;
    predictionProgressStageId = "preparing";
    predictionProgressPercent = 0;
    predictionProgressCompletedStages = new Set();
    resetPredictionUsage();
    setPredictionProgressStage("preparing", "active", 0);
    updatePredictionProgressClock();
  }
}

function startPredictionProgress() {
  stopPredictionProgress(true);
  predictionProgressStartedAt = Date.now();
  updatePredictionProgressClock();
  predictionProgressInterval = window.setInterval(updatePredictionProgressClock, 250);
}

function applyFunctionalPredictionStreamEvent(event) {
  if (event.type === "progress") {
    setPredictionProgressStage(event.stage, event.state, event.percent);
  } else if (event.type === "usage") {
    if (event.input_count !== undefined) {
      setPredictionUsage("input", event.input_count, event.input_exact);
    }
    if (event.output_count !== undefined) {
      setPredictionUsage("output", event.output_count, event.output_exact);
    }
    if (event.reasoning_count !== undefined) {
      setPredictionUsage("reasoning", event.reasoning_count, event.reasoning_exact);
    }
    if (event.phase) setPredictionUsagePhase(event.phase);
  }
}

async function consumeFunctionalPredictionStream(response) {
  if (!response.body) throw new Error("This browser does not support streamed responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let prediction = null;

  const processLine = (line) => {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === "complete") {
      const usage = event.prediction?.token_usage || {};
      if (usage.input_count !== undefined) setPredictionUsage("input", usage.input_count, true);
      if (usage.reasoning_count !== undefined) {
        setPredictionUsage("reasoning", usage.reasoning_count, true);
      }
      if (usage.output_count !== undefined) setPredictionUsage("output", usage.output_count, true);
      setPredictionUsagePhase("complete");
      setPredictionProgressStage("visualizing", "complete", 100);
      prediction = event.prediction;
    } else if (event.type === "error") {
      throw new Error(event.message || "Functional prediction failed");
    } else {
      applyFunctionalPredictionStreamEvent(event);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    lines.forEach(processLine);
    if (done) break;
  }
  if (buffer.trim()) processLine(buffer);
  if (!prediction) throw new Error("The prediction stream ended before completion.");
  return prediction;
}

function resetFunctionalPrediction(clearForm = false) {
  stopPredictionProgress(true);
  cancelPredictionGlassPreload();
  if (predictionGlassMipRequest) {
    predictionGlassMipRequest.resolve();
    predictionGlassMipRequest = null;
  }
  currentFunctionalPrediction = null;
  predictionFocusedRegionId = null;
  functionalPredictionRecords = [];
  configurePredictionDownload();
  predictionLoadedId = null;
  predictionGlassLoadedId = null;
  predictionViewMode = "slices";
  $("#prediction-empty").classList.remove("hidden");
  $("#prediction-loading").classList.add("hidden");
  $("#prediction-result").classList.add("hidden");
  renderPredictionResultUsage();
  $("#prediction-viewer-toolbar").classList.add("hidden");
  $("#prediction-viewer-loader").classList.add("hidden");
  $("#prediction-glass-loader").classList.add("hidden");
  $("#prediction-glass-render-loader").classList.add("hidden");
  $("#prediction-glass-mip-fallback").classList.add("hidden");
  $("#prediction-glass-render-fallback").classList.add("hidden");
  const glassImage = $("#prediction-glass-image");
  glassImage.classList.remove("visible");
  glassImage.onload = null;
  glassImage.onerror = null;
  glassImage.removeAttribute("src");
  delete glassImage.dataset.src;
  $("#prediction-region-filter").value = "";
  $("#prediction-coordinate-readout").textContent = "MNI   0, 0, 0";
  if (clearForm) {
    $("#prediction-form").reset();
    predictionBrainPalette = "warm";
    predictionGlassPalette = "viridis";
  }
  syncPredictionPaletteControls();
  renderFunctionalPredictionHistory();
  setPredictionViewMode("slices");
  updateFunctionalPredictionAvailability();
}

function syncPredictionPaletteControls() {
  document.querySelectorAll("[data-prediction-brain-palette]").forEach((button) => {
    const selected = button.dataset.predictionBrainPalette === predictionBrainPalette;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  document.querySelectorAll("[data-prediction-glass-palette]").forEach((button) => {
    const selected = button.dataset.predictionGlassPalette === predictionGlassPalette;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function predictionResultVolumes(prediction) {
  return [
    mniDisplayVolume("MNI152_prediction_background.nii.gz"),
    {
      url: `/api/jobs/${currentResultJob}/files/outputs/t1_to_mni_Warped.nii.gz`,
      name: "Registered_T1.nii.gz",
      colormap: "gray",
      opacity: 0,
    },
    {
      url: `/api/jobs/${currentResultJob}/files/outputs/mean_bold_mni.nii.gz`,
      name: "Registered_mean_BOLD.nii.gz",
      colormap: "gray",
      opacity: 0,
    },
    {
      url: prediction.map_url,
      name: "Predicted_Target_Task_Z_MNI152_1mm.nii.gz",
      colormap: predictionBrainPalette,
      opacity: predictionOverlayOpacity(),
      cal_min: predictionThreshold(),
      cal_max: predictionScoreMaximum(),
      trustCalMinMax: true,
      colorbarVisible: true,
    },
  ];
}

function updatePredictionViewerDisplay() {
  if (!predictionViewerAttached || predictionViewer.volumes.length < 4) return;
  const selectedBackground = Math.max(
    0,
    Math.min(2, Number($("#prediction-background-select").value)),
  );
  for (let index = 0; index < 3; index += 1) {
    predictionViewer.setOpacity(index, index === selectedBackground ? 1 : 0);
  }
  const overlay = predictionViewer.volumes[3];
  overlay.cal_min = predictionThreshold();
  overlay.cal_max = predictionScoreMaximum();
  if (overlay.colormap !== predictionBrainPalette) {
    predictionViewer.setColormap(overlay.id, predictionBrainPalette);
  }
  predictionViewer.setOpacity(3, predictionOverlayOpacity());
  predictionViewer.updateGLVolume();
  predictionViewer.drawScene();
}

async function loadPredictionViewer(prediction) {
  const loader = $("#prediction-viewer-loader");
  if (!viewerAvailable || predictionViewerUnavailable) {
    showToast("Brain Viewer requires WebGL2; the Glass Brain remains available.", true);
    return setPredictionViewMode("glass");
  }
  loader.classList.remove("hidden");
  try {
    if (!predictionViewerAttached) {
      await predictionViewer.attachToCanvas($("#prediction-brain-canvas"));
      predictionViewerAttached = true;
    }
    if (predictionLoadedId !== prediction.prediction_id) {
      await predictionViewer.loadVolumes(predictionResultVolumes(prediction));
      predictionLoadedId = prediction.prediction_id;
      predictionViewer.setSliceType(SLICE_TYPE.MULTIPLANAR);
      predictionViewer.setScale(INITIAL_3D_SCALE);
    }
    updatePredictionViewerDisplay();
    window.requestAnimationFrame(() => {
      predictionViewer.resizeListener();
      predictionViewer.drawScene();
    });
  } catch (error) {
    predictionViewerUnavailable = true;
    showToast(`Prediction Brain Viewer unavailable: ${error.message}`, true);
    return setPredictionViewMode("glass");
  } finally {
    loader.classList.add("hidden");
  }
}

function predictionGlassVolumes(prediction) {
  return [
    mniRenderVolume(),
    {
      url: prediction.glass_map_url || prediction.map_url,
      name: prediction.glass_map_url
        ? "Predicted_Target_Task_Z_3D_MNI152_2mm.nii.gz"
        : "Predicted_Target_Task_Z_3D_MNI152_1mm.nii.gz",
      colormap: glass3DPaletteName(predictionGlassPalette),
      opacity: predictionOverlayOpacity(),
      cal_min: predictionThreshold(),
      cal_max: predictionScoreMaximum(),
      trustCalMinMax: true,
    },
  ];
}

function updatePredictionGlassDisplayControls() {
  if (!predictionGlassRendererAttached || !currentFunctionalPrediction) return;
  if (predictionGlassLoadedId !== currentFunctionalPrediction.prediction_id) return;
  if (predictionGlassRenderer.volumes.length < 2) return;
  const overlay = predictionGlassRenderer.volumes[1];
  const palette = glass3DPaletteName(predictionGlassPalette);
  predictionGlassRenderer.setOpacity(0, 0.24);
  overlay.cal_min = predictionThreshold();
  overlay.cal_max = predictionScoreMaximum();
  if (overlay.colormap !== palette) {
    predictionGlassRenderer.setColormap(overlay.id, palette);
  }
  predictionGlassRenderer.setOpacity(1, predictionOverlayOpacity());
  predictionGlassRenderer.updateGLVolume();
  predictionGlassRenderer.drawScene();
}

async function loadPredictionGlassRenderer({ quiet = false } = {}) {
  const loader = $("#prediction-glass-render-loader");
  const fallback = $("#prediction-glass-render-fallback");
  if (!currentFunctionalPrediction) return;
  if (!viewerAvailable || predictionGlassRendererUnavailable) {
    loader.classList.add("hidden");
    fallback.classList.remove("hidden");
    return;
  }
  if (predictionGlassLoadPromise) {
    try {
      await predictionGlassLoadPromise;
    } catch {
      loader.classList.add("hidden");
      fallback.classList.remove("hidden");
      return;
    }
  }
  if (predictionGlassLoadedId === currentFunctionalPrediction.prediction_id) {
    fallback.classList.add("hidden");
    updatePredictionGlassDisplayControls();
    window.requestAnimationFrame(() => {
      predictionGlassRenderer.resizeListener();
      predictionGlassRenderer.drawScene();
    });
    return;
  }

  loader.classList.remove("hidden");
  fallback.classList.add("hidden");
  const prediction = currentFunctionalPrediction;
  predictionGlassLoadPromise = (async () => {
    if (!predictionGlassRendererAttached) {
      await predictionGlassRenderer.attachToCanvas($("#prediction-glass-render-canvas"));
      predictionGlassRendererAttached = true;
    }
    await predictionGlassRenderer.loadVolumes(predictionGlassVolumes(prediction));
    predictionGlassLoadedId = prediction.prediction_id;
    predictionGlassRenderer.setSliceType(SLICE_TYPE.RENDER);
    predictionGlassRenderer.setRenderAzimuthElevation(-55, 22);
    predictionGlassRenderer.setScale(INITIAL_3D_SCALE);
    updatePredictionGlassDisplayControls();
  })();
  try {
    await predictionGlassLoadPromise;
  } catch (error) {
    predictionGlassRendererUnavailable = true;
    fallback.classList.remove("hidden");
    if (!quiet) showToast(`Prediction interactive 3D unavailable: ${error.message}`, true);
  } finally {
    predictionGlassLoadPromise = null;
    loader.classList.add("hidden");
  }
}

function loadPredictionGlassMips({ quiet = false } = {}) {
  if (!currentFunctionalPrediction) return Promise.resolve();
  const image = $("#prediction-glass-image");
  const loader = $("#prediction-glass-loader");
  const fallback = $("#prediction-glass-mip-fallback");
  const query = new URLSearchParams({
    threshold: String(predictionThreshold()),
    layout: "mips",
    palette: predictionGlassPalette,
    opacity: String($("#prediction-opacity").value),
    style: "controls-v12",
  });
  const url = `${currentFunctionalPrediction.glass_brain_url}?${query}`;
  if (image.dataset.src === url && image.classList.contains("visible")) {
    return Promise.resolve();
  }
  if (predictionGlassMipRequest?.url === url) {
    predictionGlassMipRequest.reportErrors ||= !quiet;
    return predictionGlassMipRequest.promise;
  }
  if (predictionGlassMipRequest) {
    predictionGlassMipRequest.resolve();
    predictionGlassMipRequest = null;
  }
  image.classList.remove("visible");
  loader.classList.remove("hidden");
  fallback.classList.add("hidden");
  let resolveRequest;
  const promise = new Promise((resolve) => {
    resolveRequest = resolve;
  });
  const mipRequest = {
    url,
    promise,
    resolve: resolveRequest,
    reportErrors: !quiet,
  };
  predictionGlassMipRequest = mipRequest;
  const finish = () => {
    if (predictionGlassMipRequest === mipRequest) predictionGlassMipRequest = null;
    mipRequest.resolve();
  };
  image.onload = () => {
    if (image.dataset.src !== url) return finish();
    image.classList.add("visible");
    loader.classList.add("hidden");
    finish();
  };
  image.onerror = () => {
    if (image.dataset.src !== url) return finish();
    loader.classList.add("hidden");
    fallback.classList.remove("hidden");
    if (mipRequest.reportErrors) {
      showToast("Could not build the prediction Glass Brain.", true);
    }
    finish();
  };
  image.dataset.src = url;
  image.src = url;
  return promise;
}

async function loadPredictionGlassBrain(options = {}) {
  await Promise.all([
    loadPredictionGlassMips(options),
    loadPredictionGlassRenderer(options),
  ]);
}

function cancelPredictionGlassPreload() {
  if (predictionGlassPreloadHandle === null) return;
  if (predictionGlassPreloadUsesIdleCallback && typeof window.cancelIdleCallback === "function") {
    window.cancelIdleCallback(predictionGlassPreloadHandle);
  } else {
    window.clearTimeout(predictionGlassPreloadHandle);
  }
  predictionGlassPreloadHandle = null;
}

function schedulePredictionGlassPreload(predictionId) {
  cancelPredictionGlassPreload();
  const preload = () => {
    predictionGlassPreloadHandle = null;
    if (predictionViewMode !== "slices"
      || currentFunctionalPrediction?.prediction_id !== predictionId) return;
    void loadPredictionGlassBrain({ quiet: true });
  };
  predictionGlassPreloadUsesIdleCallback = typeof window.requestIdleCallback === "function";
  predictionGlassPreloadHandle = predictionGlassPreloadUsesIdleCallback
    ? window.requestIdleCallback(preload, { timeout: 750 })
    : window.setTimeout(preload, 0);
}

function setPredictionBrainPalette(palette) {
  if (!["warm", "viridis", "cool"].includes(palette)) return;
  predictionBrainPalette = palette;
  syncPredictionPaletteControls();
  updatePredictionViewerDisplay();
}

function setPredictionGlassPalette(palette) {
  if (!GLASS_3D_PALETTES.includes(palette)) return;
  predictionGlassPalette = palette;
  syncPredictionPaletteControls();
  updatePredictionGlassDisplayControls();
  if (currentFunctionalPrediction && predictionViewMode === "glass") {
    loadPredictionGlassMips();
  }
}

function setPredictionViewMode(mode) {
  predictionViewMode = mode === "glass" ? "glass" : "slices";
  const showGlass = predictionViewMode === "glass";
  $("#prediction-slice-pane").classList.toggle("hidden", showGlass);
  $("#prediction-glass-pane").classList.toggle("hidden", !showGlass);
  $("#prediction-slices-mode").classList.toggle("active", !showGlass);
  $("#prediction-glass-mode").classList.toggle("active", showGlass);
  $("#prediction-slices-mode").setAttribute("aria-pressed", String(!showGlass));
  $("#prediction-glass-mode").setAttribute("aria-pressed", String(showGlass));
  document.querySelectorAll(".prediction-slice-only-control").forEach((control) => {
    control.classList.toggle("hidden", showGlass);
  });
  document.querySelectorAll(".prediction-glass-only-control").forEach((control) => {
    control.classList.toggle("hidden", !showGlass);
  });
  if (showGlass) {
    cancelPredictionGlassPreload();
    return loadPredictionGlassBrain();
  } else if (currentFunctionalPrediction) {
    return loadPredictionViewer(currentFunctionalPrediction);
  }
  return Promise.resolve();
}

function setPredictionCrosshairMni(mni) {
  if (!predictionViewerAttached || predictionViewer.volumes.length < 1) return false;
  const crosshairPosition = predictionViewer.mm2frac(mni, 0);
  if ([...crosshairPosition].some(
    (value) => !Number.isFinite(value) || value < 0 || value > 1
  )) return false;
  predictionViewer.scene.crosshairPos = crosshairPosition;
  predictionViewer.createOnLocationChange();
  predictionViewer.drawScene();
  return true;
}

function selectPredictionRegionRow(regionId) {
  predictionFocusedRegionId = Number(regionId);
  document.querySelectorAll(".prediction-region-row").forEach((row) => {
    const selected = Number(row.dataset.regionId) === predictionFocusedRegionId;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-pressed", String(selected));
  });
}

async function focusPredictionRegion(region) {
  const mni = (region.prediction_peak_mni || region.mni)?.slice(0, 3).map(Number);
  if (!mni || mni.length !== 3 || mni.some((value) => !Number.isFinite(value))) {
    showToast("This predicted region does not have a valid MNI coordinate.", true);
    return;
  }
  if (!viewerAvailable || predictionViewerUnavailable) {
    showToast("The prediction Brain Viewer requires WebGL2.", true);
    return;
  }
  await setPredictionViewMode("slices");
  if (!setPredictionCrosshairMni(mni)) {
    showToast("This region falls outside the popup MNI volume.", true);
    return;
  }
  selectPredictionRegionRow(region.id);
  const score = Number(region.prediction_z || 0);
  showToast(`${region.name} · prediction Z ${score.toFixed(2)}`);
}

function renderPredictionRegions() {
  const list = $("#prediction-region-list");
  const regions = [...(currentFunctionalPrediction?.ranked_regions || [])].sort((left, right) => (
    Number(right.prediction_z || 0) - Number(left.prediction_z || 0)
    || Number(right.loading || 0) - Number(left.loading || 0)
  ));
  const query = $("#prediction-region-filter").value.trim().toLocaleLowerCase();
  const matches = regions.filter((region) => (
    !query || region.name.toLocaleLowerCase().includes(query)
  ));
  const visible = matches.slice(0, 36);
  list.replaceChildren(...visible.map((region) => {
    const score = Number(
      region.prediction_z ?? 0,
    );
    const peakMni = region.prediction_peak_mni || region.mni || [];
    const row = document.createElement("button");
    row.type = "button";
    row.className = "prediction-region-row";
    row.dataset.regionId = String(region.id);
    const selected = Number(region.id) === predictionFocusedRegionId;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-pressed", String(selected));
    row.setAttribute(
      "aria-label",
      `Locate ${region.name}, prediction Z ${score.toFixed(2)}, in the popup Brain Viewer`,
    );
    row.dataset.tooltip = [
      `Canonical Schaefer ID: ${region.id}`,
      `Prediction Z: ${score.toFixed(2)}`,
      `Predicted peak MNI: ${peakMni.join(", ")}`,
    ].join(" · ");
    const name = document.createElement("strong");
    name.className = "prediction-region-name";
    name.textContent = region.name;
    const bar = document.createElement("span");
    bar.className = "prediction-region-bar";
    const fill = document.createElement("i");
    fill.style.width = `${Math.max(
      0,
      Math.min(100, score / predictionScoreMaximum() * 100),
    )}%`;
    bar.appendChild(fill);
    const loading = document.createElement("b");
    loading.className = "prediction-region-score";
    loading.textContent = score.toFixed(2);
    const scoreLine = document.createElement("span");
    scoreLine.className = "prediction-region-score-line";
    scoreLine.append(bar, loading);
    row.append(name, scoreLine);
    row.addEventListener("click", () => {
      void focusPredictionRegion(region);
    });
    return row;
  }));
  window.requestAnimationFrame(() => fitRegionNames(list));
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "prediction-region-empty";
    empty.textContent = "No atlas regions match this filter.";
    list.appendChild(empty);
  }
  const activeCount = currentFunctionalPrediction?.active_regions?.length || 0;
  const regionCount = regions.length;
  const activeCutoff = Number(currentFunctionalPrediction?.active_z_cutoff ?? 0.67);
  $("#prediction-region-count").textContent = [
    query
      ? `${visible.length} of ${matches.length} matches`
      : `Top ${visible.length} of ${regionCount} regions`,
    `${activeCount} with Z ≥ ${activeCutoff.toFixed(2)}`,
  ].join(" · ");
}

async function renderFunctionalPrediction(prediction) {
  currentFunctionalPrediction = prediction;
  predictionFocusedRegionId = null;
  configurePredictionDownload();
  renderPredictionResultUsage(prediction.token_usage);
  $("#prediction-empty").classList.add("hidden");
  $("#prediction-result").classList.remove("hidden");
  $("#prediction-viewer-toolbar").classList.remove("hidden");
  $("#prediction-region-filter").value = "";
  renderPredictionRegions();
  renderFunctionalPredictionHistory();
  await setPredictionViewMode("slices");
  $("#prediction-loading").classList.add("hidden");
  schedulePredictionGlassPreload(prediction.prediction_id);
}

marked.setOptions({ gfm: true, breaks: true });

function renderMarkdown(markdown) {
  const container = document.createElement("div");
  container.className = "assistant-markdown";
  container.innerHTML = DOMPurify.sanitize(marked.parse(markdown || ""));
  container.querySelectorAll("a").forEach((link) => {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  });
  return container;
}

function svgNode(tag, attributes = {}, text = "") {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    node.setAttribute(name, String(value));
  }
  if (text) node.textContent = text;
  return node;
}

function renderTaskLabels(labels = []) {
  const values = (Array.isArray(labels) ? labels.map((item) => {
    if (typeof item === "string") return { label: item };
    return {
      label: item?.label || item?.name || "",
      confidence: Number(item?.confidence),
      evidence: item?.evidence || "",
    };
  }).filter((item) => item.label) : []).slice(0, 5);
  const card = $("#task-recognition-card");
  const labelContainer = $("#task-labels");
  card.classList.toggle("hidden", values.length === 0);
  labelContainer.dataset.count = String(values.length);
  labelContainer.replaceChildren(...values.map((item) => {
    const label = document.createElement("span");
    label.textContent = item.label;
    const confidence = Number.isFinite(item.confidence)
      ? `${Math.round(item.confidence * 100)}% confidence`
      : "";
    const tooltip = [confidence, item.evidence].filter(Boolean).join(" · ");
    if (tooltip) {
      label.dataset.tooltip = tooltip;
      label.setAttribute("aria-label", `${item.label}. ${tooltip}`);
    }
    return label;
  }));
}

function clearBoldChart() {
  $("#bold-grid").replaceChildren();
  $("#bold-series").replaceChildren();
  $("#bold-hover-mark").classList.add("hidden");
  $("#bold-tooltip").classList.add("hidden");
  boldPlotState = null;
}

function showBoldEmpty(message = "Select an ROI or move the MNI crosshair") {
  clearBoldChart();
  $("#bold-empty strong").textContent = message;
  $("#bold-empty").classList.remove("hidden");
  $("#bold-loading").classList.add("hidden");
  $("#bold-region-title").textContent = "Selected MNI region";
  $("#bold-meta").textContent = "Frame 0 →";
}

function resetBoldPlot() {
  if (boldRequestTimer) window.clearTimeout(boldRequestTimer);
  boldRequestTimer = null;
  boldRequestController?.abort();
  boldRequestController = null;
  showBoldEmpty();
}

function niceBoldLimit(values) {
  const maximum = Math.max(
    0.1,
    ...values.filter(Number.isFinite).map((value) => Math.abs(value)),
  );
  const magnitude = 10 ** Math.floor(Math.log10(maximum));
  const scaled = maximum / magnitude;
  const rounded = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return rounded * magnitude;
}

function renderBoldPlot(payload) {
  const values = (payload.values || []).map(Number);
  if (values.length < 2 || values.some((value) => !Number.isFinite(value))) {
    showBoldEmpty("No usable BOLD curve for this region");
    return;
  }
  const chart = $("#bold-chart");
  const chartBounds = chart.getBoundingClientRect();
  if (chartBounds.width > 0 && chartBounds.height > 0) {
    BOLD_PLOT.width = Math.round(chartBounds.width);
    BOLD_PLOT.height = Math.round(chartBounds.height);
    chart.setAttribute("viewBox", `0 0 ${BOLD_PLOT.width} ${BOLD_PLOT.height}`);
  }
  const grid = $("#bold-grid");
  const series = $("#bold-series");
  grid.replaceChildren();
  series.replaceChildren();
  const plotRight = BOLD_PLOT.width - BOLD_PLOT.right;
  const plotBottom = BOLD_PLOT.height - BOLD_PLOT.bottom;
  const plotWidth = plotRight - BOLD_PLOT.left;
  const plotHeight = plotBottom - BOLD_PLOT.top;
  const limit = niceBoldLimit(values);
  const xAt = (frame) => BOLD_PLOT.left
    + (frame / Math.max(1, values.length - 1)) * plotWidth;
  const yAt = (value) => BOLD_PLOT.top
    + ((limit - value) / (2 * limit)) * plotHeight;
  const zeroY = yAt(0);

  const discarded = Math.max(0, Math.min(Number(payload.discarded_frames || 0), values.length));
  if (discarded > 0) {
    const discardRight = xAt(Math.min(discarded, values.length - 1));
    grid.append(svgNode("rect", {
      x: BOLD_PLOT.left,
      y: BOLD_PLOT.top,
      width: Math.max(0, discardRight - BOLD_PLOT.left),
      height: plotHeight,
      class: "bold-discard-zone",
    }));
  }

  for (const factor of [-1, -0.5, 0, 0.5, 1]) {
    const value = factor * limit;
    const y = yAt(value);
    grid.append(
      svgNode("line", {
        x1: BOLD_PLOT.left,
        x2: plotRight,
        y1: y,
        y2: y,
        class: factor === 0 ? "bold-zero-line" : "bold-grid-line",
      }),
      svgNode("text", {
        x: BOLD_PLOT.left - 6,
        y: y + 4,
        "text-anchor": "end",
        class: "bold-axis-label",
      }, `${value.toFixed(limit < 1 ? 2 : 1)}%`),
    );
  }

  const xTicks = [...new Set([0, 0.25, 0.5, 0.75, 1].map(
    (fraction) => Math.round(fraction * (values.length - 1)),
  ))];
  for (const frame of xTicks) {
    const x = xAt(frame);
    grid.append(
      svgNode("line", {
        x1: x,
        x2: x,
        y1: BOLD_PLOT.top,
        y2: plotBottom,
        class: "bold-grid-line",
      }),
      svgNode("text", {
        x,
        y: plotBottom + 15,
        "text-anchor": "middle",
        class: "bold-axis-label",
      }, String(frame)),
    );
  }
  grid.append(
    svgNode("text", {
      x: BOLD_PLOT.left + plotWidth / 2,
      y: BOLD_PLOT.height - 4,
      "text-anchor": "middle",
      class: "bold-axis-title",
    }, "Frame number"),
    svgNode("text", {
      x: 12,
      y: BOLD_PLOT.top + plotHeight / 2,
      transform: `rotate(-90 12 ${BOLD_PLOT.top + plotHeight / 2})`,
      "text-anchor": "middle",
      class: "bold-axis-title",
    }, "% BOLD"),
  );

  const points = values.map((value, frame) => `${xAt(frame)},${yAt(value)}`);
  const linePath = `M${points.join(" L")}`;
  const areaPath = `M${xAt(0)},${zeroY} L${points.join(" L")} L${xAt(values.length - 1)},${zeroY} Z`;
  series.append(
    svgNode("path", { d: areaPath, class: "bold-area" }),
    svgNode("path", { d: linePath, class: "bold-line-shadow" }),
    svgNode("path", { d: linePath, class: "bold-line" }),
  );

  boldPlotState = {
    payload,
    values,
    xAt,
    yAt,
    plotRight,
    plotBottom,
  };
  $("#bold-empty").classList.add("hidden");
  $("#bold-loading").classList.add("hidden");
  $("#bold-region-title").textContent = payload.region || "Selected MNI region";
  $("#bold-meta").textContent = `${values.length} frames · ${payload.voxel_count} voxels · TR ${payload.tr_seconds}s`;
  $("#bold-chart").setAttribute(
    "aria-label",
    `${payload.region}, ${values.length} frames of percent BOLD signal change`,
  );
}

function boldRequestKey(jobId, mni, region = null) {
  if (Number.isInteger(Number(region?.id))) return `${jobId}:region:${region.id}`;
  const coordinate = mni.map((value) => Math.round(Number(value) * 2) / 2);
  return `${jobId}:mni:${coordinate.join(",")}`;
}

async function loadBoldTimecourse(mni, region = null) {
  if (!resultLoaded || !currentResultJob || mni.length !== 3) return;
  const coordinate = mni.map(Number);
  if (coordinate.some((value) => !Number.isFinite(value))) return;
  const jobId = currentResultJob;
  const key = boldRequestKey(jobId, coordinate, region);
  const cached = boldSeriesCache.get(key)
    || (region?.id ? boldSeriesCache.get(`${jobId}:region:${region.id}`) : null);
  boldRequestController?.abort();
  if (cached) {
    renderBoldPlot({ ...cached, mni: coordinate });
    return;
  }

  const controller = new AbortController();
  boldRequestController = controller;
  $("#bold-empty").classList.add("hidden");
  $("#bold-loading").classList.remove("hidden");
  const query = new URLSearchParams({
    x: String(coordinate[0]),
    y: String(coordinate[1]),
    z: String(coordinate[2]),
  });
  if (region?.id) query.set("region", String(region.id));
  try {
    const response = await fetch(`/api/jobs/${jobId}/bold-timeseries?${query}`, {
      signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) {
      if (response.status === 422) {
        showBoldEmpty(payload.message || "Select a highlighted cortical ROI");
        return;
      }
      throw new Error(payload.message || "Regional BOLD curve failed");
    }
    if (jobId !== currentResultJob || controller.signal.aborted) return;
    payload.mni = coordinate;
    boldSeriesCache.set(key, payload);
    boldSeriesCache.set(`${jobId}:region:${payload.region_id}`, payload);
    while (boldSeriesCache.size > 300) {
      boldSeriesCache.delete(boldSeriesCache.keys().next().value);
    }
    renderBoldPlot(payload);
  } catch (error) {
    if (error.name !== "AbortError") {
      showBoldEmpty(error.message || "Regional BOLD curve unavailable");
    }
  } finally {
    if (boldRequestController === controller) {
      boldRequestController = null;
      $("#bold-loading").classList.add("hidden");
    }
  }
}

function scheduleBoldTimecourse(mni, region = null, immediate = false) {
  if (!resultLoaded || !currentResultJob) return;
  if (boldRequestTimer) window.clearTimeout(boldRequestTimer);
  boldRequestTimer = window.setTimeout(() => {
    boldRequestTimer = null;
    loadBoldTimecourse(mni, region);
  }, immediate ? 0 : 240);
}

function setCrosshairMni(mni) {
  if (!viewerAvailable) return false;
  const crosshairPosition = slices.mm2frac(mni, 0);
  if ([...crosshairPosition].some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    return false;
  }
  slices.scene.crosshairPos = crosshairPosition;
  slices.createOnLocationChange();
  slices.drawScene();
  return true;
}

function updateBoldHover(event) {
  if (!boldPlotState) return;
  const svg = $("#bold-chart");
  const bounds = svg.getBoundingClientRect();
  const svgX = ((event.clientX - bounds.left) / bounds.width) * BOLD_PLOT.width;
  if (svgX < BOLD_PLOT.left || svgX > boldPlotState.plotRight) {
    $("#bold-hover-mark").classList.add("hidden");
    $("#bold-tooltip").classList.add("hidden");
    return;
  }
  const frame = Math.max(0, Math.min(
    boldPlotState.values.length - 1,
    Math.round(
      ((svgX - BOLD_PLOT.left) / (boldPlotState.plotRight - BOLD_PLOT.left))
      * (boldPlotState.values.length - 1),
    ),
  ));
  const value = boldPlotState.values[frame];
  const x = boldPlotState.xAt(frame);
  const y = boldPlotState.yAt(value);
  const mark = $("#bold-hover-mark");
  const line = $("#bold-hover-line");
  const dot = $("#bold-hover-dot");
  line.setAttribute("x1", x);
  line.setAttribute("x2", x);
  line.setAttribute("y1", BOLD_PLOT.top);
  line.setAttribute("y2", boldPlotState.plotBottom);
  dot.setAttribute("cx", x);
  dot.setAttribute("cy", y);
  mark.classList.remove("hidden");
  const tooltip = $("#bold-tooltip");
  tooltip.textContent = `Frame: ${frame} · BOLD Change: ${value >= 0 ? "+" : ""}${value.toFixed(3)}%`;
  tooltip.classList.remove("hidden");
  const stageWidth = $("#bold-plot-stage").clientWidth;
  const halfWidth = tooltip.offsetWidth / 2;
  const rawLeft = (x / BOLD_PLOT.width) * stageWidth;
  tooltip.style.left = `${Math.max(
    halfWidth + 8,
    Math.min(stageWidth - halfWidth - 8, rawLeft),
  )}px`;
  tooltip.style.top = `${(Math.max(y, 38) / BOLD_PLOT.height) * 100}%`;
}

function hideThinking() {
  $("#chat-thinking")?.remove();
}

function showThinking(message = "Understanding this question") {
  hideThinking();
  const list = $("#chat-list");
  list.querySelector(".chat-empty")?.remove();
  const thinking = document.createElement("div");
  thinking.id = "chat-thinking";
  thinking.className = "chat-message assistant thinking-message";
  const label = document.createElement("span");
  label.textContent = message;
  thinking.append(label);
  list.appendChild(thinking);
  list.scrollTop = list.scrollHeight;
}

function showScopeNotice(message) {
  $("#scope-notice-message").textContent = message
    || "Ask about the current fMRI result, neuroimaging, neuroscience, or relevant medical and clinical research.";
  $("#scope-notice").classList.remove("hidden");
  window.requestAnimationFrame(() => $("#scope-notice-dismiss").focus());
}

function hideScopeNotice() {
  $("#scope-notice").classList.add("hidden");
}

function appendPendingUser(question) {
  const list = $("#chat-list");
  list.querySelector(".chat-empty")?.remove();
  const user = document.createElement("div");
  user.className = "chat-message user";
  user.textContent = question;
  list.appendChild(user);
  list.scrollTop = list.scrollHeight;
  return user;
}

function createStreamingAnswer() {
  hideThinking();
  const list = $("#chat-list");
  const assistant = document.createElement("div");
  assistant.className = "chat-message assistant streaming";
  const markdown = document.createElement("div");
  markdown.className = "assistant-markdown";
  assistant.appendChild(markdown);
  list.appendChild(assistant);
  return { assistant, markdown };
}

async function consumeChatStream(response) {
  if (!response.body) throw new Error("This browser does not support streamed responses.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let rendered = null;
  let completed = false;
  let refused = false;

  const processLine = (line) => {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === "status") {
      showThinking(event.status === "screening"
        ? "Understanding this question"
        : "Analyzing the NeuroTaskFM result");
    } else if (event.type === "delta") {
      if (!rendered) rendered = createStreamingAnswer();
      answer += event.delta || "";
      rendered.markdown.innerHTML = renderMarkdown(answer).innerHTML;
      const list = $("#chat-list");
      list.scrollTop = list.scrollHeight;
    } else if (event.type === "done") {
      hideThinking();
      if (!rendered) rendered = createStreamingAnswer();
      rendered.markdown.innerHTML = renderMarkdown(event.message?.answer || answer).innerHTML;
      rendered.assistant.classList.remove("streaming");
      const timestamp = document.createElement("small");
      timestamp.textContent = formatDate(event.message?.created_at);
      rendered.assistant.appendChild(timestamp);
      completed = true;
    } else if (event.type === "refusal") {
      hideThinking();
      rendered?.assistant.remove();
      showScopeNotice(event.message);
      refused = true;
      completed = true;
    } else if (event.type === "error") {
      throw new Error(event.message || "NeuroTaskFM response failed");
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    lines.forEach(processLine);
    if (done) break;
  }
  if (buffer.trim()) processLine(buffer);
  if (!completed) throw new Error("The streamed response ended before completion.");
  return { refused };
}

function renderChat(messages) {
  const list = $("#chat-list");
  list.innerHTML = "";
  if (!messages.length) {
    list.innerHTML = '<div class="chat-empty"><span>✦</span><p>Ask what task or network pattern the result may resemble.</p></div>';
    return;
  }
  for (const item of messages) {
    const user = document.createElement("div");
    user.className = "chat-message user";
    user.textContent = item.question;
    const assistant = document.createElement("div");
    assistant.className = "chat-message assistant";
    assistant.appendChild(renderMarkdown(item.answer));
    const timestamp = document.createElement("small");
    timestamp.textContent = formatDate(item.created_at);
    assistant.appendChild(timestamp);
    list.append(user, assistant);
  }
  list.scrollTop = list.scrollHeight;
}

async function loadChat(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}/chat`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "Chat unavailable");
    llmConfigured = Boolean(payload.configured);
    for (const selector of ["#assistant-question", "#assistant-submit"]) {
      $(selector).disabled = !llmConfigured;
    }
    renderChat(payload.messages || []);
    if (!llmConfigured) resetAssistant("Configure a virtual endpoint to enable interpretation.");
  } catch (error) {
    resetAssistant(error.message);
    llmConfigured = false;
  } finally {
    updateFunctionalPredictionAvailability();
  }
}

function loadGlassBrain(jobId) {
  const image = $("#glass-brain-image");
  const loader = $("#glass-loader");
  const query = new URLSearchParams({
    threshold: String(selectedThreshold()),
    map: mapMode,
    layout: "mips",
    palette: glassPalette,
    opacity: String(opacityInput.value),
    style: "controls-v10",
  });
  const url = `/api/jobs/${jobId}/glass-brain?${query}`;
  if (viewerMode !== "glass") {
    image.dataset.pendingSrc = url;
    return;
  }
  $("#glass-layout").classList.remove("hidden");
  $("#glass-placeholder").classList.add("hidden");
  loadGlassRenderer(jobId, currentResult || {});
  if (image.dataset.src === url && image.classList.contains("visible")) return;
  image.classList.remove("visible");
  $("#glass-mip-fallback").classList.add("hidden");
  loader.classList.remove("hidden");
  image.onload = () => {
    image.classList.add("visible");
    loader.classList.add("hidden");
  };
  image.onerror = () => {
    loader.classList.add("hidden");
    $("#glass-mip-fallback").classList.remove("hidden");
    showToast("Could not load the glass-brain projections", true);
  };
  image.dataset.src = url;
  image.src = url;
}

function setGlassPalette(palette) {
  if (!["warm", "plasma", "viridis", "inferno", "turbo"].includes(palette)) return;
  glassPalette = palette;
  document.querySelectorAll("[data-palette]").forEach((button) => {
    const selected = button.dataset.palette === glassPalette;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
  updateGlassDisplayControls();
  if (resultLoaded && currentResultJob) loadGlassBrain(currentResultJob);
}

function syncBrainPaletteControls() {
  document.querySelectorAll("[data-brain-palette]").forEach((button) => {
    const selected = button.dataset.brainPalette === brainPalette;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function setBrainPalette(palette) {
  if (!["warm", "viridis", "cool"].includes(palette)) return;
  brainPalette = palette;
  syncBrainPaletteControls();
  updateDisplayControls();
}

function setViewerMode(mode) {
  viewerMode = mode === "glass" ? "glass" : "slices";
  const showGlass = viewerMode === "glass";
  $("#slice-pane").classList.toggle("hidden", showGlass);
  $("#glass-pane").classList.toggle("hidden", !showGlass);
  $("#slices-mode").classList.toggle("active", !showGlass);
  $("#glass-mode").classList.toggle("active", showGlass);
  $(".viewer-toolbar").classList.toggle("glass-mode", showGlass);
  document.querySelectorAll(".slice-only-control").forEach((control) => {
    control.classList.toggle("hidden", showGlass);
  });
  document.querySelectorAll(".glass-only-control").forEach((control) => {
    control.classList.toggle("hidden", !showGlass);
  });
  if (showGlass && currentResultJob) loadGlassBrain(currentResultJob);
  if (showGlass && glassViewerAttached) {
    window.requestAnimationFrame(() => {
      glassRenderer.resizeListener();
      glassRenderer.drawScene();
    });
  }
  if (!showGlass && viewerAvailable) {
    window.requestAnimationFrame(() => {
      slices.resizeListener();
      slices.drawScene();
    });
  }
}

function setMapMode(mode) {
  const requested = mode === "activity" ? "activity" : "dynamic";
  if (requested === "activity" && currentResult && !currentResult.outputs?.activity_mni) {
    showToast("Re-run this saved analysis to calculate the All Active Voxels map.", true);
    return;
  }
  mapMode = requested;
  brainPalette = DEFAULT_BRAIN_PALETTES[mapMode];
  syncBrainPaletteControls();
  $("#bold-card").dataset.mapMode = mapMode;
  $("#download-button").dataset.mapMode = mapMode;
  $("#dynamic-map-mode").classList.toggle("active", mapMode === "dynamic");
  $("#activity-map-mode").classList.toggle("active", mapMode === "activity");
  $("#view-dynamic-regions").classList.toggle("active", mapMode === "dynamic");
  $("#view-activity-regions").classList.toggle("active", mapMode === "activity");
  $("#dynamic-sensitivity-control").classList.toggle("hidden", mapMode !== "dynamic");
  $("#activity-threshold-control").classList.toggle("hidden", mapMode !== "activity");
  if (resultLoaded) {
    updateDisplayControls();
    renderCurrentRegions();
    configureDownload();
    if (currentResultJob) loadGlassBrain(currentResultJob);
  }
}

async function loadReference() {
  setViewerMode("slices");
  resetBoldPlot();
  renderTaskLabels();
  if (viewerAvailable) {
    const referenceVolumes = () => [mniDisplayVolume()];
    await loadSliceVolumes(referenceVolumes);
  }
  resetGlassBrain();
  resetAssistant();
  resultLoaded = false;
  currentResult = null;
  resetFunctionalPrediction(true);
  $("#activity-map-mode").disabled = true;
  $("#view-activity-regions").disabled = true;
  setMapMode("dynamic");
}

function resultVolumes(jobId, result = {}) {
  const hasRegionalDisplay = Boolean(result.outputs?.display_loading_mni);
  const displayMap = hasRegionalDisplay ? "display_loading_mni.nii.gz" : "loading_mni.nii.gz";
  const volumes = [
    mniDisplayVolume("MNI152_1mm_display.nii.gz"),
    {
      url: `/api/jobs/${jobId}/files/outputs/t1_to_mni_Warped.nii.gz`,
      name: "Registered_T1.nii.gz",
      colormap: "gray",
      opacity: 0,
    },
    {
      url: `/api/jobs/${jobId}/files/outputs/mean_bold_mni.nii.gz`,
      name: "Registered_mean_BOLD.nii.gz",
      colormap: "gray",
      opacity: 0,
    },
    {
      url: `/api/jobs/${jobId}/files/outputs/${displayMap}`,
      name: hasRegionalDisplay
        ? "Dynamic_Task_ROI.nii.gz"
        : "Dynamic_Task_ROI_voxels.nii.gz",
      colormap: brainPalette,
      opacity: Number(opacityInput.value) / 100,
      cal_min: dynamicThreshold(),
      cal_max: 100,
      trustCalMinMax: true,
      colorbarVisible: true,
    },
  ];
  if (result.outputs?.activity_mni) {
    volumes.push({
      url: `/api/jobs/${jobId}/files/outputs/activity_mni.nii.gz`,
      name: "All_Active_Voxels.nii.gz",
      colormap: brainPalette,
      opacity: 0,
      cal_min: activityThreshold(),
      cal_max: 100,
      trustCalMinMax: true,
      colorbarVisible: true,
    });
  }
  return volumes;
}

async function loadResults(jobId, result, pushHistory = true) {
  $("#open-functional-prediction").disabled = true;
  if (currentResultJob !== jobId) resetFunctionalPrediction(true);
  if (viewerAvailable) $("#slice-loader").classList.remove("hidden");
  $("#glass-loader").classList.remove("hidden");
  if (viewerAvailable) {
    await loadSliceVolumes(() => resultVolumes(jobId, result));
    slices.setSliceType(SLICE_TYPE.MULTIPLANAR);
  }
  resultLoaded = true;
  currentResultJob = jobId;
  currentResult = result;
  renderTaskLabels(result.task_labels || result.task_type_labels || []);
  const hasActivityMap = Boolean(result.outputs?.activity_mni);
  $("#activity-map-mode").disabled = !hasActivityMap;
  $("#view-activity-regions").disabled = !hasActivityMap;
  if (!hasActivityMap && mapMode === "activity") mapMode = "dynamic";
  if (viewerAvailable) $("#slice-loader").classList.add("hidden");
  setMapMode(mapMode);
  const initialRegion = result.regions?.[0] || result.activity_regions?.[0];
  if (initialRegion?.mni && (
    !viewerAvailable || setCrosshairMni(initialRegion.mni.map(Number))
  )) {
    selectRegionRow("#region-list", 0);
    scheduleBoldTimecourse(
      initialRegion.mni.map(Number),
      initialRegion,
      true,
    );
  } else {
    showBoldEmpty("Select an ROI or move the MNI crosshair");
  }
  loadGlassBrain(jobId);
  await loadChat(jobId);
  const download = $("#download-button");
  download.disabled = false;
  configureDownload();
  $("#back-button").classList.remove("hidden");
  if (pushHistory && window.location.hash !== `#result=${jobId}`) {
    window.history.pushState({ jobId }, "", `#result=${jobId}`);
  }
}

function updateDisplayControls() {
  const dynamicCutoff = dynamicThreshold();
  if (!resultLoaded || !viewerAvailable) return;
  const selected = Number(backgroundSelect.value);
  slices.setOpacity(0, selected === 0 ? 1 : 0);
  slices.setOpacity(1, selected === 1 ? 1 : 0);
  slices.setOpacity(2, selected === 2 ? 1 : 0);
  const overlayOpacity = Number(opacityInput.value) / 100;
  slices.setOpacity(3, mapMode === "dynamic" ? overlayOpacity : 0);
  slices.volumes[3].cal_min = dynamicCutoff;
  slices.volumes[3].cal_max = 100;
  if (slices.volumes[3].colormap !== brainPalette) {
    slices.setColormap(slices.volumes[3].id, brainPalette);
  }
  if (slices.volumes[4]) {
    slices.setOpacity(4, mapMode === "activity" ? overlayOpacity : 0);
    slices.volumes[4].cal_min = activityThreshold();
    slices.volumes[4].cal_max = 100;
    if (slices.volumes[4].colormap !== brainPalette) {
      slices.setColormap(slices.volumes[4].id, brainPalette);
    }
  }
  slices.updateGLVolume();
  slices.drawScene();
  updateGlassDisplayControls();
}

function configureDownload() {
  if (!currentResultJob || !currentResult) return;
  const download = $("#download-button");
  download.onclick = () => {
    const selectedMap = mapMode === "activity" ? "activity" : "dynamic";
    window.location.href = `/api/jobs/${currentResultJob}/download-map?map=${selectedMap}`;
  };
}

function configurePredictionDownload() {
  const download = $("#prediction-download-button");
  const predictionId = currentFunctionalPrediction?.prediction_id;
  if (!currentResultJob || !predictionId) {
    download.disabled = true;
    download.onclick = null;
    return;
  }
  download.disabled = false;
  download.onclick = () => {
    window.location.href = (
      `/api/jobs/${currentResultJob}/functional-predictions/`
      + `${predictionId}/download-map`
    );
  };
}

function renderCurrentRegions() {
  if (!currentResult) return;
  populateRegions("#region-list", currentResult.regions, false);
  populateRegions(
    "#activity-region-list",
    currentResult.activity_regions,
    true,
  );
}

function selectRegionRow(target, regionIndex) {
  document.querySelectorAll(".region-row.selected").forEach((row) => {
    row.classList.remove("selected");
    row.setAttribute("aria-pressed", "false");
  });
  const focusedRow = document.querySelector(`${target} .region-row[data-region-index="${regionIndex}"]`);
  if (focusedRow) {
    focusedRow.classList.add("selected");
    focusedRow.setAttribute("aria-pressed", "true");
  }
}

function focusRegion(target, regionIndex, region, activity) {
  const mni = region.mni?.slice(0, 3).map(Number);
  if (!mni || mni.length !== 3 || mni.some((value) => !Number.isFinite(value))) {
    showToast("This region does not have a valid MNI coordinate.", true);
    return;
  }

  setMapMode(activity ? "activity" : "dynamic");
  if (viewerAvailable) {
    setViewerMode("slices");
    if (!setCrosshairMni(mni)) {
      showToast("This region falls outside the displayed MNI volume.", true);
      return;
    }
  }
  selectRegionRow(target, regionIndex);
  scheduleBoldTimecourse(mni, region, true);
  const coherentArea = Number(region.coherent_support_percent || 0);
  showToast(
    `${region.name} · ROI score ${region.score ?? 0}/100 · ${coherentArea}% coherent area`,
  );
}

function fitRegionName(name, maximumSize = 14, minimumSize = 9) {
  name.style.fontSize = `${maximumSize}px`;
  const availableWidth = name.clientWidth;
  const requiredWidth = name.scrollWidth;
  if (availableWidth > 0 && requiredWidth > availableWidth) {
    const fittedSize = Math.max(
      minimumSize,
      Math.floor((maximumSize * availableWidth / requiredWidth) * 10) / 10,
    );
    name.style.fontSize = `${fittedSize}px`;
  }
}

function fitRegionNames(list) {
  list.querySelectorAll(".region-name strong").forEach((name) => fitRegionName(name));
  list.querySelectorAll(".prediction-region-name").forEach((name) => {
    fitRegionName(name, 17, 10);
  });
}

function populateRegions(target, regions = [], activity = false) {
  const list = $(target);
  list.innerHTML = "";
  list.classList.toggle("has-regions", regions.length > 0);
  const requestedLimit = Number(list.dataset.regionLimit || 20);
  const resultLimit = [8, 12, 16, 20, 24].includes(requestedLimit)
    ? requestedLimit
    : 20;
  list.style.setProperty("--region-rows", String(Math.ceil(resultLimit / 2)));
  regions.slice(0, resultLimit).forEach((region, index) => {
    const row = document.createElement("button");
    row.className = "region-row";
    row.type = "button";
    row.dataset.regionIndex = String(index);
    row.setAttribute("aria-pressed", "false");
    row.setAttribute(
      "aria-label",
      `Focus ${region.name}, ROI score ${region.score} out of 100, in the MRI viewer`,
    );
    row.dataset.tooltip = [
      `ROI Score: ${region.score}/100`,
      `Parcel Mean: ${region.mean}/100`,
      `High-Signal 90th Percentile: ${region.p90}/100`,
      `Spatially Coherent Area: ${region.coherent_support_percent}%`,
    ].join(" · ");

    const name = document.createElement("span");
    name.className = "region-name";
    const nameText = document.createElement("strong");
    nameText.textContent = region.name;
    name.appendChild(nameText);

    const scoreValue = Number(region.score ?? region.mean ?? 0);
    const score = document.createElement("span");
    score.className = "region-score";
    score.textContent = String(region.score ?? region.mean ?? 0);

    const progress = document.createElement("span");
    progress.className = "region-score-progress";
    const progressFill = document.createElement("i");
    progressFill.style.width = `${Math.max(0, Math.min(100, scoreValue))}%`;
    progress.appendChild(progressFill);

    const scoreLine = document.createElement("span");
    scoreLine.className = "region-score-line";
    scoreLine.append(progress, score);

    row.append(name, scoreLine);
    row.addEventListener("click", () => focusRegion(target, index, region, activity));
    list.appendChild(row);
  });
  window.requestAnimationFrame(() => fitRegionNames(list));
}

function updateProgress(state) {
  const progress = Number(state.progress || 0);
  $("#progress-stage").textContent = state.stage || "Working";
  $("#progress-message").textContent = state.message || "";
  $("#progress-number").textContent = `${progress}%`;
  $("#progress-bar").style.width = `${progress}%`;
  $("#progress-card").classList.toggle("is-active", state.status === "running");
}

function showToast(message, isError = false) {
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 4200);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  return bytes > 1073741824
    ? `${(bytes / 1073741824).toFixed(1)} GB`
    : `${(bytes / 1048576).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "Saved";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString([], {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
    hour12: false,
    hourCycle: "h23",
  });
}

function formatRuntime(value) {
  const totalSeconds = Math.max(0, Math.round(Number(value)));
  if (!Number.isFinite(totalSeconds)) return "";
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function matchingSavedResult() {
  if (!selectedPair) return null;
  const selectedOptions = {
    scanType: $("#scan-type").value,
    latentSources: Number($("#components").value),
    discard: Number($("#discard").value),
    regressGlobal: $("#regress-global").checked,
  };
  return resultRecords.find((result) => {
    const options = result.options || {};
    return result.pair_id === selectedPair.pair_id
      && String(options.scan_type || "task") === selectedOptions.scanType
      && Number(options.latent_sources ?? options.components ?? 5) === selectedOptions.latentSources
      && Number(options.discard ?? 5) === selectedOptions.discard
      && Boolean(options.regress_global) === selectedOptions.regressGlobal;
  }) || null;
}

function syncDuplicateAnalysisState() {
  const existing = activeJob ? null : matchingSavedResult();
  analyzeButton.classList.toggle("has-existing-result", Boolean(existing));
  if (activeJob) return;
  analyzeButton.disabled = Boolean(existing);
  analyzeButton.dataset.tooltip = existing
    ? "These Inputs and Analysis Parameters Already Have a Saved Result"
    : analyzeButtonTooltip;
  analyzeButton.querySelector("span").textContent = existing
    ? "Result already exists"
    : selectedPair ? "Analyze selected pair" : "Run latent analysis";
}

function setLibraryTab(tab) {
  const isDatasets = tab === "datasets";
  $("#datasets-tab").classList.toggle("active", isDatasets);
  $("#results-tab").classList.toggle("active", !isDatasets);
  $("#pair-list").classList.toggle("hidden", !isDatasets);
  $("#history-list").classList.toggle("hidden", isDatasets);
}

function renderPairLibrary() {
  const list = $("#pair-list");
  list.innerHTML = "";
  if (!pairRecords.length) {
    list.innerHTML = '<div class="library-empty">No saved datasets yet.</div>';
    return;
  }
  for (const pair of pairRecords) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `pair-item${selectedPair?.pair_id === pair.pair_id ? " selected" : ""}`;
    button.dataset.tooltip = `Open Dataset Pair: ${pair.label}`;
    const title = document.createElement("span");
    title.className = "pair-title";
    title.textContent = pair.label;
    const meta = document.createElement("span");
    meta.className = "pair-meta";
    const activeJobSummary = pair.active_job;
    meta.textContent = activeJobSummary
      ? `${activeJobSummary.stage || "Running"} · ${activeJobSummary.message || "Analysis in progress"}`
      : `${formatBytes(pair.uploads.fmri.size_bytes)} · ${pair.result_count || 0} result${pair.result_count === 1 ? "" : "s"}`;
    button.append(title, meta);
    if (activeJobSummary || !pair.example) {
      const badge = document.createElement("span");
      badge.className = `pair-badge${activeJobSummary ? " running" : ""}`;
      badge.textContent = activeJobSummary
        ? `${Number(activeJobSummary.progress || 0)}%`
        : pair.uploads.fmri.stored_date;
      button.appendChild(badge);
    }
    button.addEventListener("click", () => selectPair(pair));
    list.appendChild(button);
  }
}

function renderResultHistory() {
  const list = $("#history-list");
  list.innerHTML = "";
  if (!resultRecords.length) {
    list.innerHTML = '<div class="library-empty">No completed results yet.</div>';
    return;
  }
  for (const result of resultRecords) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "history-item";
    button.dataset.tooltip = `Open Saved Result: ${result.label}`;
    const title = document.createElement("span");
    title.className = "history-title";
    title.textContent = result.label;
    const meta = document.createElement("span");
    meta.className = "history-meta";
    const runtime = formatRuntime(
      result.pipeline_runtime_seconds ?? result.runtime_seconds,
    );
    meta.textContent = [formatDate(result.completed_at), runtime].filter(Boolean).join(" · ");
    const score = document.createElement("span");
    score.className = "history-score";
    const roiScore = result.top_regions?.[0]?.score;
    score.textContent = Number.isFinite(roiScore) ? `${roiScore}/100` : "View";
    button.append(title, meta, score);
    button.addEventListener("click", () => openSavedResult(result.job_id));
    list.appendChild(button);
  }
}

async function refreshLibrary() {
  try {
    const [pairsResponse, resultsResponse] = await Promise.all([
      fetch("/api/pairs"), fetch("/api/results"),
    ]);
    const pairs = await pairsResponse.json();
    const results = await resultsResponse.json();
    nextPairName = pairs.next_pair_name || nextPairName;
    pairRecords = pairs.pairs || [];
    resultRecords = results.results || [];
    if (selectedPair) {
      selectedPair = pairRecords.find((pair) => pair.pair_id === selectedPair.pair_id) || null;
    }
    if (!selectedPair && runNameInput.dataset.source === "auto") {
      setRunName(nextPairName);
    }
    renderPairLibrary();
    renderResultHistory();
    syncDuplicateAnalysisState();
  } catch (error) {
    showToast(`Could not refresh saved datasets: ${error.message}`, true);
  }
}

async function previewPair(pair) {
  setViewerMode("slices");
  resetBoldPlot();
  renderTaskLabels();
  if (!viewerAvailable) {
    resultLoaded = false;
    currentResult = null;
    resetFunctionalPrediction(true);
    resetGlassBrain("Analyze this pair to create its Glass Brain projections");
    resetAssistant("Analyze this pair before asking about its loading pattern.");
    return;
  }
  $("#slice-loader").classList.remove("hidden");
  $("#glass-loader").classList.remove("hidden");
  const volumes = () => [{
    url: `/api/pairs/${pair.pair_id}/files/t1.nii.gz`,
    name: `${pair.label || "Saved_T1"}.nii.gz`,
    colormap: "gray",
    opacity: 1,
  }];
  try {
    await loadSliceVolumes(volumes);
    slices.setSliceType(SLICE_TYPE.MULTIPLANAR);
    resultLoaded = false;
    currentResult = null;
    resetFunctionalPrediction(true);
    $("#activity-map-mode").disabled = true;
    $("#view-activity-regions").disabled = true;
    setMapMode("dynamic");
    resetGlassBrain("Analyze this pair to create its Glass Brain projections");
    resetAssistant("Analyze this pair before asking the LLM about its loading pattern.");
  } catch (error) {
    showToast(`Could not preview saved T1: ${error.message}`, true);
  } finally {
    $("#slice-loader").classList.add("hidden");
    $("#glass-loader").classList.add("hidden");
  }
}

async function selectPair(pair) {
  if (activeJob) {
    showToast("Wait for the current analysis to finish before changing datasets.", true);
    return;
  }
  selectedPair = pair;
  t1Input.value = "";
  fmriInput.value = "";
  $("#t1-drop").classList.remove("has-file");
  $("#fmri-drop").classList.remove("has-file");
  $("#t1-name").textContent = pair.uploads.t1.original_name;
  $("#fmri-name").textContent = pair.uploads.fmri.original_name;
  setRunName(pair.label, "selected");
  analyzeButton.querySelector("span").textContent = "Analyze selected pair";
  renderPairLibrary();
  await previewPair(pair);
  syncDuplicateAnalysisState();
  if (pair.active_job?.job_id) {
    activeJob = pair.active_job.job_id;
    activeJobWasReused = false;
    analyzeButton.disabled = true;
    runNameInput.disabled = true;
    analyzeButton.querySelector("span").textContent = "Analysis running";
    updateProgress(pair.active_job);
    if (pollTimer) window.clearInterval(pollTimer);
    await pollJob(activeJob);
    if (activeJob) {
      pollTimer = window.setInterval(() => pollJob(activeJob), 1800);
    }
  }
}

async function openSavedResult(jobId, pushHistory = true) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const state = await response.json();
    if (!response.ok || state.status !== "complete") {
      throw new Error(state.message || "Saved result is unavailable");
    }
    const options = state.options || {};
    if (["task", "rest"].includes(options.scan_type)) {
      $("#scan-type").value = options.scan_type;
    }
    const latentSources = Number(options.latent_sources ?? options.components);
    if ([3, 4, 5, 6].includes(latentSources)) {
      $("#components").value = String(latentSources);
    }
    if (Number.isFinite(Number(options.discard))) {
      $("#discard").value = String(options.discard);
    }
    $("#regress-global").checked = Boolean(options.regress_global);
    selectedPair = pairRecords.find((pair) => pair.pair_id === state.pair_id) || null;
    setRunName(state.run_label || state.pair_label || "Saved analysis", "selected");
    renderPairLibrary();
    updateProgress(state);
    await loadResults(jobId, state.result, pushHistory);
    setLibraryTab("results");
    syncDuplicateAnalysisState();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function returnToDatasets(pushHistory = true) {
  resetBoldPlot();
  currentResultJob = null;
  resultLoaded = false;
  resetFunctionalPrediction(true);
  $("#back-button").classList.add("hidden");
  setLibraryTab("datasets");
  if (pushHistory) window.history.pushState({}, "", window.location.pathname);
  if (selectedPair) await previewPair(selectedPair);
  else await loadReference();
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    const state = await response.json();
    updateProgress(state);
    if (state.status === "complete") {
      const reusedExisting = activeJobWasReused;
      activeJobWasReused = false;
      window.clearInterval(pollTimer);
      activeJob = null;
      analyzeButton.disabled = false;
      runNameInput.disabled = false;
      analyzeButton.querySelector("span").textContent = "Run another analysis";
      await loadResults(jobId, state.result, true);
      await refreshLibrary();
      if (!selectedPair) setRunName(nextPairName);
      showToast(reusedExisting
        ? "Identical inputs and parameters already exist; opened the saved result"
        : state.cache_hit
          ? "Loaded the existing cached result"
          : `Analysis completed in ${Math.round(state.result.runtime_seconds)} seconds`);
    } else if (state.status === "failed") {
      activeJobWasReused = false;
      window.clearInterval(pollTimer);
      activeJob = null;
      analyzeButton.disabled = false;
      runNameInput.disabled = false;
      analyzeButton.querySelector("span").textContent = "Retry analysis";
      showToast(state.message || "Analysis failed", true);
    }
  } catch (error) {
    showToast(`Unable to read job status: ${error.message}`, true);
  }
}

async function previewFile(file) {
  if (!file || activeJob) return;
  setViewerMode("slices");
  resetBoldPlot();
  renderTaskLabels();
  if (!viewerAvailable) {
    showToast("T1 preview requires a WebGL2-capable browser.", true);
    return;
  }
  $("#slice-loader").classList.remove("hidden");
  $("#glass-loader").classList.remove("hidden");
  try {
    while (slices.volumes.length) slices.removeVolumeByIndex(0);
    await slices.loadFromFile(file);
    slices.setSliceType(SLICE_TYPE.MULTIPLANAR);
    resultLoaded = false;
    currentResult = null;
    resetFunctionalPrediction(true);
    $("#activity-map-mode").disabled = true;
    $("#view-activity-regions").disabled = true;
    setMapMode("dynamic");
    resetGlassBrain("Complete the analysis to create the Glass Brain projections");
  } catch (error) {
    showToast(`Could not preview ${file.name}: ${error.message}`, true);
    await loadReference();
  } finally {
    $("#slice-loader").classList.add("hidden");
    $("#glass-loader").classList.add("hidden");
  }
}

function bindFileInput(input, nameTarget, card, preview = false) {
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    $(nameTarget).textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    $(card).classList.add("has-file");
    selectedPair = null;
    if (runNameInput.dataset.source !== "user") setRunName(nextPairName);
    renderPairLibrary();
    analyzeButton.querySelector("span").textContent = "Run latent analysis";
    syncDuplicateAnalysisState();
    if (preview) previewFile(file);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedPair && (!t1Input.files[0] || !fmriInput.files[0])) {
    showToast("Select both a T1 and a 4D fMRI file.", true);
    return;
  }
  const runName = currentRunName();
  if (!runName) {
    showToast("Enter a name for this run.", true);
    runNameInput.focus();
    return;
  }
  if (pollTimer) window.clearInterval(pollTimer);
  const options = {
    scan_type: $("#scan-type").value,
    latent_sources: Number($("#components").value),
    discard: Number($("#discard").value),
    regress_global: $("#regress-global").checked,
  };
  let url = "/api/jobs";
  let fetchOptions;
  if (selectedPair) {
    url = `/api/pairs/${selectedPair.pair_id}/jobs`;
    fetchOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...options, run_name: runName }),
    };
  } else {
    const payload = new FormData();
    payload.append("t1", t1Input.files[0]);
    payload.append("fmri", fmriInput.files[0]);
    payload.append("run_name", runName);
    for (const [key, value] of Object.entries(options)) payload.append(key, String(value));
    fetchOptions = { method: "POST", body: payload };
  }
  analyzeButton.disabled = true;
  runNameInput.disabled = true;
  analyzeButton.querySelector("span").textContent = selectedPair ? "Opening dataset…" : "Uploading scans…";
  updateProgress({
    progress: 2,
    status: "running",
    stage: selectedPair ? "Opening dataset" : "Uploading",
    message: selectedPair ? "Checking the result cache" : "Transferring and hashing NIfTI files",
  });
  try {
    const response = await fetch(url, fetchOptions);
    const state = await response.json();
    if (!response.ok) throw new Error(state.message || "Upload failed");
    activeJob = state.job_id;
    activeJobWasReused = Boolean(state.reused_existing_job);
    analyzeButton.querySelector("span").textContent = "Analysis running";
    updateProgress(state);
    pollTimer = window.setInterval(() => pollJob(activeJob), 1800);
    await pollJob(activeJob);
  } catch (error) {
    activeJobWasReused = false;
    analyzeButton.disabled = false;
    runNameInput.disabled = false;
    analyzeButton.querySelector("span").textContent = "Run latent analysis";
    showToast(error.message, true);
  }
});

[$("#scan-type"), $("#components"), $("#regress-global")].forEach((control) => {
  control.addEventListener("change", syncDuplicateAnalysisState);
});
$("#discard").addEventListener("input", syncDuplicateAnalysisState);

$("#prediction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentResultJob || !resultLoaded || !llmConfigured) {
    showToast("Open a completed result and configure the server API key first.", true);
    return;
  }
  const sourceTask = $("#prediction-source-task").value.trim();
  const contrastId = $("#prediction-target-task").value.trim();
  if (sourceTask.length < 2) {
    showToast("Briefly describe what the subject did in the measured run.", true);
    $("#prediction-source-task").focus();
    return;
  }
  if (!functionalContrastCatalogLoaded || contrastId.length < 2) {
    showToast("Choose an available target contrast.", true);
    $("#prediction-target-task").focus();
    return;
  }

  const jobId = currentResultJob;
  const button = $("#prediction-submit");
  const hadPrediction = Boolean(currentFunctionalPrediction);
  button.disabled = true;
  button.textContent = "Generating contrast map…";
  $("#prediction-empty").classList.add("hidden");
  $("#prediction-result").classList.add("hidden");
  $("#prediction-viewer-toolbar").classList.add("hidden");
  $("#prediction-loading").classList.remove("hidden");
  startPredictionProgress();
  try {
    const response = await fetch(`/api/jobs/${jobId}/functional-predictions?stream=1`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/x-ndjson",
      },
      body: JSON.stringify({
        source_task: sourceTask,
        contrast_id: contrastId,
      }),
    });
    if (!response.ok) {
      const failure = await response.json();
      throw new Error(failure.message || "Functional prediction failed");
    }
    const prediction = await consumeFunctionalPredictionStream(response);
    if (jobId !== currentResultJob) return;
    await renderFunctionalPrediction(prediction);
    void loadFunctionalPredictionHistory();
    showToast(
      `Generated ${prediction.ranked_regions?.length || 50} functional loadings in ${formatRuntime(prediction.runtime_seconds)}.`,
    );
  } catch (error) {
    $("#prediction-loading").classList.add("hidden");
    $(hadPrediction ? "#prediction-result" : "#prediction-empty").classList.remove("hidden");
    $("#prediction-viewer-toolbar").classList.toggle("hidden", !hadPrediction);
    showToast(error.message, true);
  } finally {
    stopPredictionProgress();
    button.disabled = false;
    button.textContent = "Generate prediction";
  }
});

$("#assistant-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!currentResultJob || !llmConfigured) {
    showToast("Open a result and configure the server API key first.", true);
    return;
  }
  const questionField = $("#assistant-question");
  const question = questionField.value.trim();
  if (!question) {
    showToast("Enter a question for the research assistant.", true);
    return;
  }
  const button = $("#assistant-submit");
  const payload = new FormData();
  payload.append("question", question);
  payload.append("dynamic_threshold", String(dynamicThreshold()));
  payload.append("activity_threshold", String(activityThreshold()));
  questionField.value = "";
  button.disabled = true;
  button.textContent = "Thinking";
  hideScopeNotice();
  const pendingUser = appendPendingUser(question);
  showThinking("Understanding this question");
  try {
    const response = await fetch(`/api/jobs/${currentResultJob}/chat`, {
      method: "POST",
      body: payload,
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.message || "NeuroTaskFM request failed");
    }
    const outcome = await consumeChatStream(response);
    if (outcome.refused) {
      pendingUser.remove();
      if (!$("#chat-list").children.length) renderChat([]);
    }
  } catch (error) {
    hideThinking();
    showToast(error.message, true);
  } finally {
    button.disabled = !llmConfigured;
    button.textContent = "Send";
  }
});

thresholdInput.addEventListener("change", () => {
  updateDisplayControls();
  if (resultLoaded && currentResultJob && mapMode === "activity") loadGlassBrain(currentResultJob);
});
dynamicSensitivityInput.addEventListener("change", () => {
  updateDisplayControls();
  if (resultLoaded && currentResultJob && mapMode === "dynamic") loadGlassBrain(currentResultJob);
});
opacityInput.addEventListener("change", () => {
  updateDisplayControls();
  if (resultLoaded && currentResultJob && viewerMode === "glass") {
    loadGlassBrain(currentResultJob);
  }
});
backgroundSelect.addEventListener("change", updateDisplayControls);
document.querySelectorAll("[data-palette]").forEach((button) => {
  button.addEventListener("click", () => setGlassPalette(button.dataset.palette));
});
document.querySelectorAll("[data-brain-palette]").forEach((button) => {
  button.addEventListener("click", () => setBrainPalette(button.dataset.brainPalette));
});
$("#slices-mode").addEventListener("click", () => setViewerMode("slices"));
$("#glass-mode").addEventListener("click", () => setViewerMode("glass"));
$("#dynamic-map-mode").addEventListener("click", () => setMapMode("dynamic"));
$("#activity-map-mode").addEventListener("click", () => setMapMode("activity"));
$("#view-dynamic-regions").addEventListener("click", () => setMapMode("dynamic"));
$("#view-activity-regions").addEventListener("click", () => setMapMode("activity"));
$("#dynamic-region-limit").addEventListener("change", (event) => {
  $("#region-list").dataset.regionLimit = event.target.value;
  renderCurrentRegions();
});
$("#activity-region-limit").addEventListener("change", (event) => {
  $("#activity-region-list").dataset.regionLimit = event.target.value;
  renderCurrentRegions();
});
$("#open-functional-prediction").addEventListener("click", () => {
  if (!functionalContrastCatalogLoaded) loadFunctionalContrastCatalog();
  void loadFunctionalPredictionHistory();
  const dialog = $("#prediction-dialog");
  if (!dialog.open) dialog.showModal();
  window.requestAnimationFrame(() => {
    if (currentFunctionalPrediction && predictionViewMode === "slices") {
      predictionViewer.resizeListener();
      predictionViewer.drawScene();
    }
    if (
      currentFunctionalPrediction
      && predictionViewMode === "glass"
      && predictionGlassRendererAttached
    ) {
      predictionGlassRenderer.resizeListener();
      predictionGlassRenderer.drawScene();
    }
    $("#prediction-source-task").focus();
  });
});
$("#close-functional-prediction").addEventListener("click", () => {
  $("#prediction-dialog").close();
});
$("#prediction-dialog").addEventListener("click", (event) => {
  const dialog = event.currentTarget;
  const bounds = dialog.getBoundingClientRect();
  const inside = event.clientX >= bounds.left && event.clientX <= bounds.right
    && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
  if (!inside) dialog.close();
});
$("#prediction-dialog").addEventListener("close", hideAppTooltip);
$("#prediction-slices-mode").addEventListener("click", () => setPredictionViewMode("slices"));
$("#prediction-glass-mode").addEventListener("click", () => setPredictionViewMode("glass"));
$("#prediction-background-select").addEventListener("change", updatePredictionViewerDisplay);
$("#prediction-sensitivity").addEventListener("change", () => {
  if (!currentFunctionalPrediction) return;
  updatePredictionViewerDisplay();
  updatePredictionGlassDisplayControls();
  if (predictionViewMode === "glass") loadPredictionGlassMips();
  else schedulePredictionGlassPreload(currentFunctionalPrediction.prediction_id);
});
$("#prediction-opacity").addEventListener("change", () => {
  if (!currentFunctionalPrediction) return;
  updatePredictionViewerDisplay();
  updatePredictionGlassDisplayControls();
  if (predictionViewMode === "glass") loadPredictionGlassMips();
  else schedulePredictionGlassPreload(currentFunctionalPrediction.prediction_id);
});
document.querySelectorAll("[data-prediction-brain-palette]").forEach((button) => {
  button.addEventListener("click", () => (
    setPredictionBrainPalette(button.dataset.predictionBrainPalette)
  ));
});
document.querySelectorAll("[data-prediction-glass-palette]").forEach((button) => {
  button.addEventListener("click", () => (
    setPredictionGlassPalette(button.dataset.predictionGlassPalette)
  ));
});
$("#prediction-region-filter").addEventListener("input", renderPredictionRegions);
$("#open-assistant").addEventListener("click", () => {
  const dialog = $("#assistant-dialog");
  if (!dialog.open) dialog.showModal();
  if (!$("#assistant-question").disabled) {
    window.requestAnimationFrame(() => $("#assistant-question").focus());
  }
});
$("#close-assistant").addEventListener("click", () => {
  hideScopeNotice();
  $("#assistant-dialog").close();
});
$("#scope-notice-dismiss").addEventListener("click", hideScopeNotice);
$("#scope-notice").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) hideScopeNotice();
});
$("#assistant-dialog").addEventListener("cancel", (event) => {
  if (!$("#scope-notice").classList.contains("hidden")) {
    event.preventDefault();
    hideScopeNotice();
  }
});
$("#assistant-dialog").addEventListener("click", (event) => {
  const dialog = event.currentTarget;
  const bounds = dialog.getBoundingClientRect();
  const inside = event.clientX >= bounds.left && event.clientX <= bounds.right
    && event.clientY >= bounds.top && event.clientY <= bounds.bottom;
  if (!inside) dialog.close();
});
$("#assistant-dialog").addEventListener("close", hideAppTooltip);
$("#bold-chart").addEventListener("pointermove", updateBoldHover);
$("#bold-chart").addEventListener("pointerleave", () => {
  $("#bold-hover-mark").classList.add("hidden");
  $("#bold-tooltip").classList.add("hidden");
});
new ResizeObserver(([entry]) => {
  if (!boldPlotState?.payload || !entry) return;
  const width = Math.round(entry.contentRect.width);
  const height = Math.round(entry.contentRect.height);
  if (width === BOLD_PLOT.width && height === BOLD_PLOT.height) return;
  if (boldResizeFrame) window.cancelAnimationFrame(boldResizeFrame);
  boldResizeFrame = window.requestAnimationFrame(() => {
    boldResizeFrame = null;
    renderBoldPlot(boldPlotState.payload);
  });
}).observe($("#bold-chart"));
const regionNameObserver = new ResizeObserver((entries) => {
  entries.forEach((entry) => fitRegionNames(entry.target));
});
regionNameObserver.observe($("#region-list"));
regionNameObserver.observe($("#activity-region-list"));
regionNameObserver.observe($("#prediction-region-list"));
runNameInput.addEventListener("input", () => {
  runNameInput.dataset.source = "user";
});
bindFileInput(t1Input, "#t1-name", "#t1-drop", true);
bindFileInput(fmriInput, "#fmri-name", "#fmri-drop", false);
$("#datasets-tab").addEventListener("click", () => setLibraryTab("datasets"));
$("#results-tab").addEventListener("click", () => setLibraryTab("results"));
$("#refresh-library").addEventListener("click", refreshLibrary);
$("#back-button").addEventListener("click", () => returnToDatasets());
document.addEventListener("pointerover", (event) => {
  if (event.pointerType === "touch") return;
  const target = tooltipTarget(event);
  if (!target || target === appTooltipTarget) return;
  showAppTooltip(target);
});
document.addEventListener("pointerout", (event) => {
  const target = tooltipTarget(event);
  if (!target || target !== appTooltipTarget) return;
  if (event.relatedTarget instanceof Node && target.contains(event.relatedTarget)) return;
  hideAppTooltip();
});
document.addEventListener("focusin", (event) => {
  const target = tooltipTarget(event);
  if (target) showAppTooltip(target);
});
document.addEventListener("focusout", (event) => {
  const target = tooltipTarget(event);
  if (!target || target !== appTooltipTarget) return;
  if (event.relatedTarget instanceof Node && target.contains(event.relatedTarget)) return;
  hideAppTooltip();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") hideAppTooltip();
});
document.addEventListener("scroll", hideAppTooltip, true);
window.addEventListener("resize", hideAppTooltip);
window.addEventListener("popstate", async () => {
  const resultId = new URLSearchParams(window.location.hash.slice(1)).get("result");
  if (resultId) await openSavedResult(resultId, false);
  else await returnToDatasets(false);
});

fetch("/api/health")
  .then((response) => response.json())
  .then((health) => {
    llmConfigured = Boolean(health.llm_configured);
  })
  .catch(() => {
    llmConfigured = false;
  });

loadFunctionalContrastCatalog();
initializeViewers().catch((error) => showToast(`Viewer initialization failed: ${error.message}`, true));
