"""训练可视化 HTML 模板。

包含基于 Plotly.js 的交互式可视化前端模板，支持：
- 实时训练指标图表（损失、准确率等）
- 交互式参数调整面板
- 训练过程动画回放
- 图表导出为 PNG/SVG

Author: TinyAI Team
"""

VISUALIZER_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tinyTorch Training Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-card: #21253a;
    --accent: #6c63ff;
    --accent-hover: #857dff;
    --text-primary: #e8eaed;
    --text-secondary: #9aa0a6;
    --border: #2d3250;
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
  }

  /* Header */
  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .header-left { display: flex; align-items: center; gap: 16px; }
  .logo {
    font-size: 22px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .status-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .status-badge.idle { background: rgba(154,160,166,0.15); color: var(--text-secondary); }
  .status-badge.running { background: rgba(52,211,153,0.15); color: var(--success); }
  .status-badge.done { background: rgba(108,99,255,0.15); color: var(--accent); }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: currentColor;
  }
  .status-badge.running .status-dot { animation: pulse 1.5s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  .header-actions { display: flex; gap: 8px; }
  .btn {
    padding: 8px 16px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-primary);
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .btn:hover { border-color: var(--accent); background: rgba(108,99,255,0.1); }
  .btn-primary { background: var(--accent); border-color: var(--accent); }
  .btn-primary:hover { background: var(--accent-hover); }

  /* Layout */
  .main { display: flex; height: calc(100vh - 65px); }

  /* Sidebar */
  .sidebar {
    width: 320px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    flex-shrink: 0;
  }
  .sidebar-section {
    padding: 20px;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: var(--text-secondary);
    margin-bottom: 14px;
  }

  /* Stat cards */
  .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .stat-card {
    background: var(--bg-card);
    border-radius: 10px;
    padding: 14px;
    border: 1px solid var(--border);
  }
  .stat-label { font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; }
  .stat-value { font-size: 20px; font-weight: 700; }
  .stat-value.loss { color: var(--danger); }
  .stat-value.acc { color: var(--success); }
  .stat-value.epoch { color: var(--accent); }
  .stat-value.time { color: var(--warning); }

  /* Param controls */
  .param-group { margin-bottom: 16px; }
  .param-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 6px;
  }
  .param-value { color: var(--accent); font-weight: 600; }
  input[type="range"] {
    width: 100%;
    -webkit-appearance: none;
    height: 6px;
    border-radius: 3px;
    background: var(--border);
    outline: none;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 18px;
    border-radius: 50%;
    background: var(--accent);
    cursor: grab;
    border: 2px solid var(--bg-primary);
    box-shadow: 0 2px 6px rgba(108,99,255,0.4);
  }
  input[type="range"]::-webkit-slider-thumb:active { cursor: grabbing; }

  /* Animation controls */
  .anim-controls {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
  }
  .anim-btn {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-primary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    transition: all 0.2s;
  }
  .anim-btn:hover { border-color: var(--accent); background: rgba(108,99,255,0.15); }
  .anim-btn.active { background: var(--accent); border-color: var(--accent); }
  .anim-progress {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    position: relative;
    cursor: pointer;
  }
  .anim-progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s;
  }
  .anim-label { font-size: 12px; color: var(--text-secondary); min-width: 60px; text-align: right; }

  /* Content area */
  .content { flex: 1; overflow-y: auto; padding: 24px; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-card {
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
  }
  .chart-card.full-width { grid-column: 1 / -1; }
  .chart-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .chart-title { font-size: 14px; font-weight: 600; }
  .chart-actions { display: flex; gap: 6px; }
  .chart-action-btn {
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 11px;
    transition: all 0.2s;
  }
  .chart-action-btn:hover { color: var(--text-primary); border-color: var(--accent); }
  .chart-body { padding: 8px; }

  /* Toast */
  .toast-container { position: fixed; top: 20px; right: 20px; z-index: 1000; }
  .toast {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 20px;
    margin-bottom: 8px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 8px;
    animation: slideIn 0.3s ease;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
  }
  @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }

  /* Responsive */
  @media (max-width: 1200px) { .chart-grid { grid-template-columns: 1fr; } }
  @media (max-width: 900px) { .sidebar { width: 260px; } }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">&#x1F525; tinyTorch Visualizer</div>
    <div class="status-badge idle" id="statusBadge">
      <div class="status-dot"></div>
      <span id="statusText">Idle</span>
    </div>
  </div>
  <div class="header-actions">
    <a class="btn" href="/graph" style="text-decoration:none;">&#x1F50D; Computation Graph</a>
    <button class="btn" onclick="exportAllCharts('png')">&#x1F4F7; Export PNG</button>
    <button class="btn" onclick="exportAllCharts('svg')">&#x1F4BE; Export SVG</button>
    <button class="btn" onclick="saveState()">&#x1F4E5; Save State</button>
    <button class="btn btn-primary" onclick="refreshData()">&#x21BB; Refresh</button>
  </div>
</div>

<div class="main">
  <div class="sidebar">
    <!-- Stats -->
    <div class="sidebar-section">
      <div class="sidebar-title">Training Overview</div>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-label">Current Epoch</div>
          <div class="stat-value epoch" id="statEpoch">-</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Elapsed Time</div>
          <div class="stat-value time" id="statTime">-</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Train Loss</div>
          <div class="stat-value loss" id="statTrainLoss">-</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Val Loss</div>
          <div class="stat-value loss" id="statValLoss">-</div>
        </div>
      </div>
    </div>

    <!-- Display Parameters -->
    <div class="sidebar-section">
      <div class="sidebar-title">Display Controls</div>
      <div class="param-group">
        <div class="param-label">
          <span>Smoothing</span>
          <span class="param-value" id="smoothVal">0.0</span>
        </div>
        <input type="range" id="smoothSlider" min="0" max="0.99" step="0.01" value="0"
               oninput="onSmoothChange(this.value)">
      </div>
      <div class="param-group">
        <div class="param-label">
          <span>Y-Axis Scale</span>
          <span class="param-value" id="scaleVal">Linear</span>
        </div>
        <input type="range" id="scaleSlider" min="0" max="1" step="1" value="0"
               oninput="onScaleChange(this.value)">
      </div>
      <div class="param-group">
        <div class="param-label">
          <span>Display Range (epochs)</span>
          <span class="param-value" id="rangeVal">All</span>
        </div>
        <input type="range" id="rangeSlider" min="0" max="100" step="1" value="0"
               oninput="onRangeChange(this.value)">
      </div>
    </div>

    <!-- Animation -->
    <div class="sidebar-section">
      <div class="sidebar-title">Animation Playback</div>
      <p style="font-size:12px; color:var(--text-secondary); margin-bottom:10px;">
        Replay the training process step by step
      </p>
      <div class="anim-controls">
        <button class="anim-btn" id="animResetBtn" onclick="animReset()" title="Reset">&#x23EE;</button>
        <button class="anim-btn" id="animPlayBtn" onclick="animToggle()" title="Play/Pause">&#x25B6;</button>
        <div class="anim-progress" onclick="animSeek(event)">
          <div class="anim-progress-fill" id="animProgressFill" style="width:0%"></div>
        </div>
        <div class="anim-label" id="animLabel">0 / 0</div>
      </div>
      <div class="param-group" style="margin-top:14px;">
        <div class="param-label">
          <span>Playback Speed</span>
          <span class="param-value" id="speedVal">1x</span>
        </div>
        <input type="range" id="speedSlider" min="0.25" max="4" step="0.25" value="1"
               oninput="onSpeedChange(this.value)">
      </div>
    </div>
  </div>

  <div class="content">
    <div class="chart-grid">
      <div class="chart-card full-width">
        <div class="chart-header">
          <div class="chart-title">&#x1F4C9; Loss Curves</div>
          <div class="chart-actions">
            <button class="chart-action-btn" onclick="exportChart('lossChart','png')">PNG</button>
            <button class="chart-action-btn" onclick="exportChart('lossChart','svg')">SVG</button>
          </div>
        </div>
        <div class="chart-body"><div id="lossChart" style="height:340px;"></div></div>
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">&#x1F4CA; Loss Distribution</div>
          <div class="chart-actions">
            <button class="chart-action-btn" onclick="exportChart('distChart','png')">PNG</button>
          </div>
        </div>
        <div class="chart-body"><div id="distChart" style="height:300px;"></div></div>
      </div>

      <div class="chart-card">
        <div class="chart-header">
          <div class="chart-title">&#x1F4C8; Loss Change Rate</div>
          <div class="chart-actions">
            <button class="chart-action-btn" onclick="exportChart('rateChart','png')">PNG</button>
          </div>
        </div>
        <div class="chart-body"><div id="rateChart" style="height:300px;"></div></div>
      </div>

      <div class="chart-card full-width" id="customMetricsCard" style="display:none;">
        <div class="chart-header">
          <div class="chart-title">&#x1F4CB; Custom Metrics</div>
          <div class="chart-actions">
            <button class="chart-action-btn" onclick="exportChart('customChart','png')">PNG</button>
          </div>
        </div>
        <div class="chart-body"><div id="customChart" style="height:300px;"></div></div>
      </div>
    </div>
  </div>
</div>

<div class="toast-container" id="toastContainer"></div>

<script>
// ========== State ==========
let trainingData = null;
let smoothingFactor = 0;
let yAxisType = 'linear';
let displayRange = 0;
let animPlaying = false;
let animFrame = 0;
let animTimer = null;
let animSpeed = 1;
const DATA_URL = '/api/training_data';
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#9aa0a6', size: 12 },
  margin: { l: 50, r: 20, t: 10, b: 40 },
  xaxis: {
    gridcolor: 'rgba(45,50,80,0.5)',
    zerolinecolor: 'rgba(45,50,80,0.5)',
    title: { text: 'Epoch', font: { size: 12 } }
  },
  yaxis: {
    gridcolor: 'rgba(45,50,80,0.5)',
    zerolinecolor: 'rgba(45,50,80,0.5)',
  },
  legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center', font: { size: 11 } },
  hovermode: 'x unified',
};
const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

// ========== Data Loading ==========
async function refreshData() {
  try {
    const resp = await fetch(DATA_URL);
    if (!resp.ok) throw new Error('Failed to fetch');
    trainingData = await resp.json();
    updateAll();
    setStatus(trainingData.status || 'done');
    showToast('Data refreshed successfully', 'var(--success)');
  } catch (err) {
    showToast('Failed to load data: ' + err.message, 'var(--danger)');
  }
}

function updateAll() {
  if (!trainingData) return;
  updateStats();
  updateLossChart();
  updateDistChart();
  updateRateChart();
  updateCustomChart();
  updateAnimRange();
}

// ========== Stats ==========
function updateStats() {
  const d = trainingData;
  const trainLoss = d.train_losses || [];
  const valLoss = d.val_losses || [];
  document.getElementById('statEpoch').textContent = d.current_epoch || trainLoss.length || '-';
  document.getElementById('statTime').textContent = d.elapsed_time ? formatTime(d.elapsed_time) : '-';
  document.getElementById('statTrainLoss').textContent = trainLoss.length ? trainLoss[trainLoss.length-1].toFixed(4) : '-';
  document.getElementById('statValLoss').textContent = valLoss.length ? valLoss[valLoss.length-1].toFixed(4) : '-';
}

function formatTime(seconds) {
  if (seconds < 60) return seconds.toFixed(1) + 's';
  if (seconds < 3600) return (seconds/60).toFixed(1) + 'm';
  return (seconds/3600).toFixed(1) + 'h';
}

// ========== Smoothing ==========
function exponentialSmoothing(values, factor) {
  if (factor <= 0 || !values.length) return values;
  const result = [values[0]];
  for (let i = 1; i < values.length; i++) {
    result.push(factor * result[i-1] + (1 - factor) * values[i]);
  }
  return result;
}

function getDisplayData(values) {
  let data = exponentialSmoothing(values, smoothingFactor);
  if (displayRange > 0 && data.length > displayRange) {
    data = data.slice(data.length - displayRange);
  }
  return data;
}

function getEpochAxis(length) {
  const offset = (displayRange > 0 && length > displayRange) ? length - displayRange : 0;
  const count = (displayRange > 0 && length > displayRange) ? displayRange : length;
  return Array.from({length: count}, (_, i) => i + offset + 1);
}

// ========== Loss Chart ==========
function updateLossChart() {
  const trainLoss = trainingData.train_losses || [];
  const valLoss = trainingData.val_losses || [];
  if (!trainLoss.length) return;

  const displayTrain = getDisplayData(trainLoss);
  const epochs = getEpochAxis(trainLoss.length);
  const traces = [];

  // Raw data (faded) when smoothing is on
  if (smoothingFactor > 0) {
    traces.push({
      x: epochs, y: displayRange > 0 ? trainLoss.slice(-displayRange) : trainLoss,
      type: 'scatter', mode: 'lines', name: 'Train (raw)',
      line: { color: 'rgba(248,113,113,0.2)', width: 1 },
      showlegend: false, hoverinfo: 'skip'
    });
  }

  traces.push({
    x: epochs, y: displayTrain,
    type: 'scatter', mode: 'lines+markers', name: 'Train Loss',
    line: { color: '#f87171', width: 2.5 },
    marker: { size: 3, color: '#f87171' },
  });

  if (valLoss.length) {
    const displayVal = getDisplayData(valLoss);
    if (smoothingFactor > 0) {
      traces.push({
        x: epochs, y: displayRange > 0 ? valLoss.slice(-displayRange) : valLoss,
        type: 'scatter', mode: 'lines', name: 'Val (raw)',
        line: { color: 'rgba(251,191,36,0.2)', width: 1 },
        showlegend: false, hoverinfo: 'skip'
      });
    }
    traces.push({
      x: epochs, y: displayVal,
      type: 'scatter', mode: 'lines+markers', name: 'Val Loss',
      line: { color: '#fbbf24', width: 2.5 },
      marker: { size: 3, color: '#fbbf24' },
    });
  }

  const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE));
  layout.yaxis.type = yAxisType;
  layout.yaxis.title = { text: 'Loss', font: { size: 12 } };

  Plotly.react('lossChart', traces, layout, PLOTLY_CONFIG);
}

// ========== Distribution Chart ==========
function updateDistChart() {
  const trainLoss = trainingData.train_losses || [];
  if (!trainLoss.length) return;

  const traces = [{
    x: trainLoss, type: 'histogram', name: 'Train Loss',
    marker: { color: 'rgba(248,113,113,0.6)', line: { color: '#f87171', width: 1 } },
    nbinsx: Math.min(30, Math.max(10, Math.floor(trainLoss.length / 3))),
  }];

  const valLoss = trainingData.val_losses || [];
  if (valLoss.length) {
    traces.push({
      x: valLoss, type: 'histogram', name: 'Val Loss',
      marker: { color: 'rgba(251,191,36,0.5)', line: { color: '#fbbf24', width: 1 } },
      nbinsx: Math.min(30, Math.max(10, Math.floor(valLoss.length / 3))),
    });
  }

  const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE));
  layout.barmode = 'overlay';
  layout.xaxis.title = { text: 'Loss Value', font: { size: 12 } };
  layout.yaxis.title = { text: 'Count', font: { size: 12 } };

  Plotly.react('distChart', traces, layout, PLOTLY_CONFIG);
}

// ========== Rate Chart ==========
function updateRateChart() {
  const trainLoss = trainingData.train_losses || [];
  if (trainLoss.length < 2) return;

  const changeRates = [];
  for (let i = 1; i < trainLoss.length; i++) {
    changeRates.push(trainLoss[i] - trainLoss[i-1]);
  }

  const epochs = Array.from({length: changeRates.length}, (_, i) => i + 2);
  const colors = changeRates.map(v => v < 0 ? 'rgba(52,211,153,0.7)' : 'rgba(248,113,113,0.7)');

  const traces = [{
    x: epochs, y: changeRates, type: 'bar', name: 'Loss Change',
    marker: { color: colors },
  }];

  const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE));
  layout.xaxis.title = { text: 'Epoch', font: { size: 12 } };
  layout.yaxis.title = { text: 'Delta Loss', font: { size: 12 } };
  layout.showlegend = false;

  Plotly.react('rateChart', traces, layout, PLOTLY_CONFIG);
}

// ========== Custom Metrics Chart ==========
function updateCustomChart() {
  const customMetrics = trainingData.custom_metrics || {};
  const metricNames = Object.keys(customMetrics);
  if (!metricNames.length) {
    document.getElementById('customMetricsCard').style.display = 'none';
    return;
  }
  document.getElementById('customMetricsCard').style.display = '';

  const palette = ['#6c63ff','#34d399','#f87171','#fbbf24','#a78bfa','#38bdf8','#fb923c'];
  const traces = metricNames.map((name, idx) => {
    const values = customMetrics[name];
    const displayValues = getDisplayData(values);
    const epochs = getEpochAxis(values.length);
    return {
      x: epochs, y: displayValues,
      type: 'scatter', mode: 'lines+markers', name: name,
      line: { color: palette[idx % palette.length], width: 2 },
      marker: { size: 3 },
    };
  });

  const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE));
  layout.yaxis.type = yAxisType;
  Plotly.react('customChart', traces, layout, PLOTLY_CONFIG);
}

// ========== Display Controls ==========
function onSmoothChange(val) {
  smoothingFactor = parseFloat(val);
  document.getElementById('smoothVal').textContent = smoothingFactor.toFixed(2);
  updateLossChart();
  updateCustomChart();
}

function onScaleChange(val) {
  yAxisType = parseInt(val) === 1 ? 'log' : 'linear';
  document.getElementById('scaleVal').textContent = yAxisType === 'log' ? 'Log' : 'Linear';
  updateLossChart();
  updateCustomChart();
}

function onRangeChange(val) {
  displayRange = parseInt(val);
  document.getElementById('rangeVal').textContent = displayRange === 0 ? 'All' : displayRange.toString();
  updateLossChart();
  updateCustomChart();
}

// ========== Animation ==========
function updateAnimRange() {
  const total = (trainingData.train_losses || []).length;
  document.getElementById('rangeSlider').max = Math.max(total, 10);
  animFrame = total;
  updateAnimUI();
}

function updateAnimUI() {
  const total = (trainingData && trainingData.train_losses) ? trainingData.train_losses.length : 0;
  const pct = total > 0 ? (animFrame / total * 100) : 0;
  document.getElementById('animProgressFill').style.width = pct + '%';
  document.getElementById('animLabel').textContent = animFrame + ' / ' + total;
}

function animToggle() {
  animPlaying = !animPlaying;
  const btn = document.getElementById('animPlayBtn');
  if (animPlaying) {
    btn.classList.add('active');
    btn.innerHTML = '&#x23F8;';
    animPlay();
  } else {
    btn.classList.remove('active');
    btn.innerHTML = '&#x25B6;';
    if (animTimer) { clearTimeout(animTimer); animTimer = null; }
  }
}

function animPlay() {
  if (!animPlaying || !trainingData) return;
  const total = trainingData.train_losses.length;
  if (animFrame >= total) {
    animPlaying = false;
    document.getElementById('animPlayBtn').classList.remove('active');
    document.getElementById('animPlayBtn').innerHTML = '&#x25B6;';
    return;
  }
  animFrame++;
  renderAnimFrame();
  updateAnimUI();
  animTimer = setTimeout(animPlay, 500 / animSpeed);
}

function animReset() {
  animPlaying = false;
  if (animTimer) { clearTimeout(animTimer); animTimer = null; }
  animFrame = 0;
  document.getElementById('animPlayBtn').classList.remove('active');
  document.getElementById('animPlayBtn').innerHTML = '&#x25B6;';
  renderAnimFrame();
  updateAnimUI();
}

function animSeek(event) {
  if (!trainingData) return;
  const bar = event.currentTarget;
  const rect = bar.getBoundingClientRect();
  const pct = (event.clientX - rect.left) / rect.width;
  const total = trainingData.train_losses.length;
  animFrame = Math.round(pct * total);
  renderAnimFrame();
  updateAnimUI();
}

function onSpeedChange(val) {
  animSpeed = parseFloat(val);
  document.getElementById('speedVal').textContent = animSpeed + 'x';
}

function renderAnimFrame() {
  if (!trainingData) return;
  const trainLoss = trainingData.train_losses.slice(0, animFrame);
  const valLoss = (trainingData.val_losses || []).slice(0, animFrame);
  const epochs = Array.from({length: animFrame}, (_, i) => i + 1);
  const totalEpochs = trainingData.train_losses.length;

  const traces = [{
    x: epochs, y: exponentialSmoothing(trainLoss, smoothingFactor),
    type: 'scatter', mode: 'lines+markers', name: 'Train Loss',
    line: { color: '#f87171', width: 2.5 }, marker: { size: 3 },
  }];

  if (valLoss.length) {
    traces.push({
      x: epochs, y: exponentialSmoothing(valLoss, smoothingFactor),
      type: 'scatter', mode: 'lines+markers', name: 'Val Loss',
      line: { color: '#fbbf24', width: 2.5 }, marker: { size: 3 },
    });
  }

  const layout = JSON.parse(JSON.stringify(PLOTLY_LAYOUT_BASE));
  layout.yaxis.type = yAxisType;
  layout.yaxis.title = { text: 'Loss', font: { size: 12 } };
  layout.xaxis.range = [0, totalEpochs + 1];

  Plotly.react('lossChart', traces, layout, PLOTLY_CONFIG);

  // Update stats for current frame
  document.getElementById('statEpoch').textContent = animFrame || '-';
  document.getElementById('statTrainLoss').textContent = trainLoss.length ? trainLoss[trainLoss.length-1].toFixed(4) : '-';
  document.getElementById('statValLoss').textContent = valLoss.length ? valLoss[valLoss.length-1].toFixed(4) : '-';
}

// ========== Export ==========
function exportChart(chartId, format) {
  Plotly.downloadImage(chartId, {
    format: format, width: 1200, height: 600,
    filename: 'tinytorch_' + chartId + '_' + new Date().toISOString().slice(0,10)
  });
  showToast('Chart exported as ' + format.toUpperCase(), 'var(--success)');
}

function exportAllCharts(format) {
  const charts = ['lossChart', 'distChart', 'rateChart'];
  if (document.getElementById('customMetricsCard').style.display !== 'none') {
    charts.push('customChart');
  }
  charts.forEach(id => exportChart(id, format));
}

function saveState() {
  if (!trainingData) { showToast('No data to save', 'var(--warning)'); return; }
  const stateJson = JSON.stringify({
    training_data: trainingData,
    display: { smoothing: smoothingFactor, yAxisType, displayRange },
    timestamp: new Date().toISOString()
  }, null, 2);
  const blob = new Blob([stateJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'tinytorch_viz_state_' + new Date().toISOString().slice(0,10) + '.json';
  anchor.click();
  URL.revokeObjectURL(url);
  showToast('State saved successfully', 'var(--success)');
}

// ========== Status ==========
function setStatus(status) {
  const badge = document.getElementById('statusBadge');
  const text = document.getElementById('statusText');
  badge.className = 'status-badge ' + status;
  const labels = { idle: 'Idle', running: 'Training...', done: 'Completed' };
  text.textContent = labels[status] || status;
}

// ========== Toast ==========
function showToast(message, color) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = '<span style="color:' + (color||'var(--text-primary)') + '">&#x25CF;</span> ' + message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.3s'; }, 2500);
  setTimeout(() => container.removeChild(toast), 3000);
}

// ========== Auto-refresh ==========
let autoRefreshTimer = null;
function startAutoRefresh(intervalMs) {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  autoRefreshTimer = setInterval(refreshData, intervalMs || 3000);
}

// ========== Init ==========
window.addEventListener('load', () => {
  refreshData();
  startAutoRefresh(3000);
});
</script>
</body>
</html>
"""
