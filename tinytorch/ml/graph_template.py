"""计算图可视化 HTML 模板。

基于 D3.js 力导向图的交互式计算图可视化前端，支持：
- 动态计算图（Tensor + Function 节点）
- 模块层级结构图
- 节点拖拽、缩放、平移
- 节点详情悬浮面板
- 图布局切换（力导向 / 层级）
- 导出为 PNG/SVG

Author: TinyAI Team
"""

GRAPH_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>tinyTorch - Computation Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
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
    overflow: hidden;
    height: 100vh;
  }

  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex; align-items: center; justify-content: space-between;
    z-index: 10; position: relative;
  }
  .header-left { display: flex; align-items: center; gap: 16px; }
  .logo {
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .tab-bar { display: flex; gap: 4px; }
  .tab {
    padding: 6px 16px; border-radius: 8px; border: 1px solid var(--border);
    background: transparent; color: var(--text-secondary); cursor: pointer;
    font-size: 13px; font-weight: 500; transition: all 0.2s;
  }
  .tab:hover { border-color: var(--accent); color: var(--text-primary); }
  .tab.active { background: var(--accent); border-color: var(--accent); color: #fff; }
  .header-actions { display: flex; gap: 8px; }
  .btn {
    padding: 6px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg-card); color: var(--text-primary); cursor: pointer;
    font-size: 12px; font-weight: 500; transition: all 0.2s;
    display: flex; align-items: center; gap: 5px; text-decoration: none;
  }
  .btn:hover { border-color: var(--accent); background: rgba(108,99,255,0.1); }

  .main { display: flex; height: calc(100vh - 53px); }
  .canvas-container { flex: 1; position: relative; }
  svg { width: 100%; height: 100%; }

  .legend {
    position: absolute; bottom: 20px; left: 20px;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px; font-size: 12px;
  }
  .legend-title { font-weight: 600; margin-bottom: 8px; color: var(--text-secondary); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .legend-item { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
  .legend-dot { width: 12px; height: 12px; border-radius: 3px; }

  .info-panel {
    width: 300px; background: var(--bg-secondary); border-left: 1px solid var(--border);
    overflow-y: auto; padding: 20px; flex-shrink: 0;
  }
  .info-panel.hidden { display: none; }
  .info-title { font-size: 16px; font-weight: 700; margin-bottom: 16px; }
  .info-section { margin-bottom: 16px; }
  .info-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-secondary); margin-bottom: 8px; }
  .info-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(45,50,80,0.3); font-size: 13px; }
  .info-key { color: var(--text-secondary); }
  .info-value { font-weight: 600; }

  .stats-bar {
    position: absolute; top: 12px; right: 320px;
    display: flex; gap: 10px;
  }
  .stats-bar.panel-hidden { right: 12px; }
  .stat-chip {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 8px; padding: 6px 14px; font-size: 12px;
    display: flex; align-items: center; gap: 6px;
  }
  .stat-chip .num { font-weight: 700; color: var(--accent); }

  .tooltip {
    position: absolute; pointer-events: none; z-index: 100;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; font-size: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    max-width: 280px; opacity: 0; transition: opacity 0.15s;
  }
  .tooltip.visible { opacity: 1; }

  .empty-state {
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    text-align: center; color: var(--text-secondary);
  }
  .empty-state .icon { font-size: 48px; margin-bottom: 12px; }
  .empty-state .msg { font-size: 14px; }

  /* Edge paths with arrows */
  .link-path { stroke: rgba(108,99,255,0.4); stroke-width: 2; fill: none; }
  .link-path:hover { stroke: var(--accent); stroke-width: 3; }
  .arrowhead { fill: rgba(108,99,255,0.6); }

  /* Nodes */
  .node { cursor: pointer; }
  .node-rect { rx: 8; ry: 8; stroke-width: 2; transition: filter 0.2s; }
  .node:hover .node-rect { filter: brightness(1.3) drop-shadow(0 0 8px rgba(108,99,255,0.4)); }
  .node-label { fill: #fff; font-size: 11px; font-weight: 600; text-anchor: middle; dominant-baseline: central; pointer-events: none; }
  .node-sublabel { fill: rgba(255,255,255,0.6); font-size: 9px; text-anchor: middle; dominant-baseline: central; pointer-events: none; }

  /* Module nesting boxes */
  .module-group-rect { rx: 12; ry: 12; stroke-width: 1.5; stroke-dasharray: 6 3; fill-opacity: 0.04; }
  .module-group-label { font-size: 10px; font-weight: 600; fill: rgba(167,139,250,0.7); }
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">&#x1F50D; Graph Visualizer</div>
    <div class="tab-bar">
      <button class="tab active" id="tabCompGraph" onclick="switchTab('computation')">Computation Graph</button>
      <button class="tab" id="tabModGraph" onclick="switchTab('module')">Module Structure</button>
    </div>
  </div>
  <div class="header-actions">
    <a class="btn" href="/">&#x1F4CA; Training Metrics</a>
    <button class="btn" onclick="resetZoom()">&#x1F3AF; Reset View</button>
    <button class="btn" onclick="exportSVG()">&#x1F4BE; Export SVG</button>
    <button class="btn" onclick="exportPNG()">&#x1F4F7; Export PNG</button>
    <button class="btn" onclick="togglePanel()">&#x2139;&#xFE0F; Info Panel</button>
  </div>
</div>

<div class="main">
  <div class="canvas-container" id="canvasContainer">
    <svg id="graphSvg"></svg>
    <div class="stats-bar" id="statsBar">
      <div class="stat-chip">Tensors: <span class="num" id="statTensors">0</span></div>
      <div class="stat-chip">Ops: <span class="num" id="statOps">0</span></div>
      <div class="stat-chip">Edges: <span class="num" id="statEdges">0</span></div>
    </div>
    <div class="legend" id="legend"></div>
    <div class="tooltip" id="tooltip"></div>
    <div class="empty-state" id="emptyState" style="display:none;">
      <div class="icon">&#x1F50D;</div>
      <div class="msg">No graph data available</div>
    </div>
  </div>
  <div class="info-panel" id="infoPanel">
    <div class="info-title" id="infoTitle">Select a node</div>
    <div id="infoContent">
      <p style="color:var(--text-secondary); font-size:13px;">Click on a node to view its details.</p>
    </div>
  </div>
</div>

<script>
let graphData = null;
let currentTab = 'computation';
let svgGroup = null;
let zoomBehavior = null;
const DATA_URL = '/api/graph_data';

function nodeColor(node) {
  if (node.type === 'function') return '#f87171';
  if (node.type === 'module') return '#a78bfa';
  if (node.subtype === 'leaf') return '#34d399';
  return '#38bdf8';
}
function nodeWidth(node) {
  if (node.type === 'module') return Math.max(120, node.label.length * 7 + 30);
  if (node.type === 'function') return Math.max(90, node.label.length * 7 + 20);
  return Math.max(100, node.label.length * 6.5 + 20);
}
function nodeHeight(node) {
  return (node.type === 'module' && node.parameters && node.parameters.length > 0) ? 48 : 36;
}

// ========== Data Loading ==========
async function loadData() {
  try {
    if (window._offlineGraphData) {
      graphData = window._offlineGraphData;
    } else {
      const resp = await fetch(DATA_URL);
      if (!resp.ok) throw new Error('Failed to fetch');
      graphData = await resp.json();
    }
    renderCurrentTab();
  } catch (err) {
    document.getElementById('emptyState').style.display = '';
  }
}

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tabCompGraph').classList.toggle('active', tab === 'computation');
  document.getElementById('tabModGraph').classList.toggle('active', tab === 'module');
  renderCurrentTab();
}

function renderCurrentTab() {
  if (!graphData) return;
  if (currentTab === 'computation') {
    const data = graphData.computation_graph;
    if (!data || !data.nodes || data.nodes.length === 0) {
      showEmpty(); return;
    }
    document.getElementById('emptyState').style.display = 'none';
    renderComputationGraph(data);
  } else {
    const data = graphData.module_graph;
    if (!data || !data.nodes || data.nodes.length === 0) {
      showEmpty(); return;
    }
    document.getElementById('emptyState').style.display = 'none';
    renderModuleGraph(data);
  }
  updateLegend();
}

function showEmpty() {
  document.getElementById('emptyState').style.display = '';
  clearGraph();
  updateStats(0, 0, 0);
}

function updateStats(a, b, c) {
  document.getElementById('statTensors').textContent = a;
  document.getElementById('statOps').textContent = b;
  document.getElementById('statEdges').textContent = c;
}

function updateLegend() {
  const legend = document.getElementById('legend');
  if (currentTab === 'computation') {
    legend.innerHTML =
      '<div class="legend-title">Node Types</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#34d399"></div> Leaf Tensor (input/param)</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#38bdf8"></div> Intermediate Tensor</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#f87171"></div> Function (operation)</div>';
  } else {
    legend.innerHTML =
      '<div class="legend-title">Node Types</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div> Module (container)</div>' +
      '<div class="legend-item"><div class="legend-dot" style="background:#6c63ff;border-radius:50%"></div> Module (leaf layer)</div>';
  }
}

function clearGraph() {
  d3.select('#graphSvg').selectAll('*').remove();
}

// ================================================================
//  COMPUTATION GRAPH — Left-to-right layered layout with arrows
// ================================================================
function renderComputationGraph(data) {
  clearGraph();
  const svg = d3.select('#graphSvg');
  const container = document.getElementById('canvasContainer');
  const cWidth = container.clientWidth;
  const cHeight = container.clientHeight;

  // Defs
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 10).attr('refY', 0)
    .attr('markerWidth', 8).attr('markerHeight', 8)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-4L8,0L0,4Z').attr('class', 'arrowhead');

  // Zoom
  zoomBehavior = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => svgGroup.attr('transform', e.transform));
  svg.call(zoomBehavior);
  svgGroup = svg.append('g');

  const nodeMap = {};
  data.nodes.forEach(n => { nodeMap[n.id] = n; });
  const validEdges = data.edges.filter(e => nodeMap[e.source] && nodeMap[e.target]);

  // Stats
  const tensorCount = data.nodes.filter(n => n.type === 'tensor').length;
  const funcCount = data.nodes.filter(n => n.type === 'function').length;
  updateStats(tensorCount, funcCount, validEdges.length);

  // ---- Layered layout (left to right by depth) ----
  const depthGroups = {};
  data.nodes.forEach(n => {
    const d = n.depth !== undefined ? n.depth : 0;
    if (!depthGroups[d]) depthGroups[d] = [];
    depthGroups[d].push(n);
  });

  const depths = Object.keys(depthGroups).map(Number).sort((a, b) => a - b);
  const layerSpacing = 200;
  const nodeSpacing = 70;

  depths.forEach(depth => {
    const group = depthGroups[depth];
    const totalHeight = group.length * nodeSpacing;
    const startY = -totalHeight / 2 + nodeSpacing / 2;
    group.forEach((node, idx) => {
      node.x = depth * layerSpacing;
      node.y = startY + idx * nodeSpacing;
    });
  });

  // Center the graph
  const allX = data.nodes.map(n => n.x);
  const allY = data.nodes.map(n => n.y);
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  const graphW = maxX - minX + 300;
  const graphH = maxY - minY + 200;
  const offsetX = (cWidth - graphW) / 2 - minX + 150;
  const offsetY = (cHeight - graphH) / 2 - minY + 100;
  data.nodes.forEach(n => { n.x += offsetX; n.y += offsetY; });

  // Draw edges as curved paths with arrows
  const linkGen = d3.linkHorizontal()
    .source(d => {
      const s = nodeMap[d.source];
      return [s.x + nodeWidth(s) / 2, s.y];
    })
    .target(d => {
      const t = nodeMap[d.target];
      return [t.x - nodeWidth(t) / 2 - 10, t.y];
    });

  svgGroup.append('g').selectAll('path')
    .data(validEdges)
    .enter().append('path')
    .attr('class', 'link-path')
    .attr('d', linkGen)
    .attr('marker-end', 'url(#arrow)');

  // Draw nodes
  const nodeGroup = svgGroup.append('g').attr('class', 'nodes');
  const nodes = nodeGroup.selectAll('g')
    .data(data.nodes)
    .enter().append('g')
    .attr('class', 'node')
    .attr('transform', d => 'translate(' + d.x + ',' + d.y + ')')
    .on('click', (ev, d) => showNodeInfo(d))
    .on('mouseenter', (ev, d) => showTooltip(ev, d))
    .on('mouseleave', () => hideTooltip());

  nodes.append('rect')
    .attr('class', 'node-rect')
    .attr('width', d => nodeWidth(d))
    .attr('height', d => nodeHeight(d))
    .attr('x', d => -nodeWidth(d) / 2)
    .attr('y', d => -nodeHeight(d) / 2)
    .attr('fill', d => nodeColor(d))
    .attr('fill-opacity', 0.15)
    .attr('stroke', d => nodeColor(d));

  nodes.append('text')
    .attr('class', 'node-label')
    .attr('y', d => nodeHeight(d) > 36 ? -6 : 0)
    .text(d => d.label);

  nodes.filter(d => d.shape)
    .append('text')
    .attr('class', 'node-sublabel')
    .attr('y', 10)
    .text(d => '[' + d.shape.join(', ') + ']');
}

// ================================================================
//  MODULE GRAPH — Hierarchical nested layout (top-down with nesting boxes)
// ================================================================
function renderModuleGraph(data) {
  clearGraph();
  const svg = d3.select('#graphSvg');
  const container = document.getElementById('canvasContainer');
  const cWidth = container.clientWidth;
  const cHeight = container.clientHeight;

  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow-mod')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 10).attr('refY', 0)
    .attr('markerWidth', 7).attr('markerHeight', 7)
    .attr('orient', 'auto')
    .append('path').attr('d', 'M0,-4L8,0L0,4Z').attr('class', 'arrowhead');

  zoomBehavior = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => svgGroup.attr('transform', e.transform));
  svg.call(zoomBehavior);
  svgGroup = svg.append('g');

  const nodeMap = {};
  data.nodes.forEach(n => { nodeMap[n.id] = n; });
  const validEdges = data.edges.filter(e => nodeMap[e.source] && nodeMap[e.target]);

  const moduleCount = data.nodes.length;
  updateStats(moduleCount, 0, validEdges.length);

  // ---- Hierarchical layout ----
  // Assign positions: root at left, children flow left-to-right by depth, top-to-bottom by order
  const layerSpacing = 220;
  const nodeSpacingY = 80;

  // Group by depth
  const depthGroups = {};
  data.nodes.forEach(n => {
    const d = n.depth !== undefined ? n.depth : 0;
    if (!depthGroups[d]) depthGroups[d] = [];
    depthGroups[d].push(n);
  });

  // Sort each depth group by order
  Object.values(depthGroups).forEach(group => {
    group.sort((a, b) => (a.order || 0) - (b.order || 0));
  });

  const depths = Object.keys(depthGroups).map(Number).sort((a, b) => a - b);

  depths.forEach(depth => {
    const group = depthGroups[depth];
    const totalH = group.length * nodeSpacingY;
    const startY = -totalH / 2 + nodeSpacingY / 2;
    group.forEach((node, idx) => {
      node.x = depth * layerSpacing;
      node.y = startY + idx * nodeSpacingY;
    });
  });

  // Center
  const allX = data.nodes.map(n => n.x);
  const allY = data.nodes.map(n => n.y);
  const minX = Math.min(...allX), maxX = Math.max(...allX);
  const minY = Math.min(...allY), maxY = Math.max(...allY);
  const graphW = maxX - minX + 400;
  const graphH = maxY - minY + 200;
  const offsetX = (cWidth - graphW) / 2 - minX + 200;
  const offsetY = (cHeight - graphH) / 2 - minY + 100;
  data.nodes.forEach(n => { n.x += offsetX; n.y += offsetY; });

  // Draw nesting boxes for container modules (those with children)
  const containers = data.nodes.filter(n => n.children && n.children.length > 0);
  const nestGroup = svgGroup.append('g').attr('class', 'nesting');

  containers.forEach(containerNode => {
    // Collect all descendant positions
    const descendantNodes = collectDescendants(containerNode.id, nodeMap);
    if (descendantNodes.length === 0) return;

    const xs = descendantNodes.map(n => n.x);
    const ys = descendantNodes.map(n => n.y);
    const padding = 40;
    const maxNw = Math.max(...descendantNodes.map(n => nodeWidth(n)));
    const maxNh = Math.max(...descendantNodes.map(n => nodeHeight(n)));
    const bx = Math.min(...xs) - maxNw / 2 - padding;
    const by = Math.min(...ys) - maxNh / 2 - padding - 16;
    const bw = Math.max(...xs) - Math.min(...xs) + maxNw + padding * 2;
    const bh = Math.max(...ys) - Math.min(...ys) + maxNh + padding * 2 + 16;

    const depthColors = ['rgba(167,139,250,0.3)', 'rgba(108,99,255,0.25)', 'rgba(56,189,248,0.2)', 'rgba(52,211,153,0.15)'];
    const strokeColor = depthColors[Math.min(containerNode.depth || 0, depthColors.length - 1)];

    nestGroup.append('rect')
      .attr('class', 'module-group-rect')
      .attr('x', bx).attr('y', by)
      .attr('width', bw).attr('height', bh)
      .attr('stroke', strokeColor)
      .attr('fill', strokeColor);

    nestGroup.append('text')
      .attr('class', 'module-group-label')
      .attr('x', bx + 8).attr('y', by + 14)
      .text(containerNode.label);
  });

  // Draw edges
  const linkGen = d3.linkHorizontal()
    .source(d => {
      const s = nodeMap[d.source];
      return [s.x + nodeWidth(s) / 2, s.y];
    })
    .target(d => {
      const t = nodeMap[d.target];
      return [t.x - nodeWidth(t) / 2 - 8, t.y];
    });

  svgGroup.append('g').selectAll('path')
    .data(validEdges)
    .enter().append('path')
    .attr('class', 'link-path')
    .attr('d', linkGen)
    .attr('marker-end', 'url(#arrow-mod)');

  // Draw nodes
  const nodeGroup = svgGroup.append('g').attr('class', 'nodes');
  const nodes = nodeGroup.selectAll('g')
    .data(data.nodes)
    .enter().append('g')
    .attr('class', 'node')
    .attr('transform', d => 'translate(' + d.x + ',' + d.y + ')')
    .on('click', (ev, d) => showNodeInfo(d))
    .on('mouseenter', (ev, d) => showTooltip(ev, d))
    .on('mouseleave', () => hideTooltip());

  nodes.append('rect')
    .attr('class', 'node-rect')
    .attr('width', d => nodeWidth(d))
    .attr('height', d => nodeHeight(d))
    .attr('x', d => -nodeWidth(d) / 2)
    .attr('y', d => -nodeHeight(d) / 2)
    .attr('fill', d => nodeColor(d))
    .attr('fill-opacity', 0.15)
    .attr('stroke', d => nodeColor(d));

  nodes.append('text')
    .attr('class', 'node-label')
    .attr('y', d => nodeHeight(d) > 36 ? -6 : 0)
    .text(d => d.label);

  nodes.filter(d => d.parameters && d.parameters.length > 0)
    .append('text')
    .attr('class', 'node-sublabel')
    .attr('y', 10)
    .text(d => d.parameters.map(function(p) { return p.name + ':' + p.shape.join('x'); }).join(', '));
}

function collectDescendants(nodeId, nodeMap) {
  const result = [];
  const stack = [nodeId];
  while (stack.length > 0) {
    const currentId = stack.pop();
    const node = nodeMap[currentId];
    if (!node) continue;
    if (node.children) {
      node.children.forEach(function(childId) {
        const child = nodeMap[childId];
        if (child) {
          result.push(child);
          stack.push(childId);
        }
      });
    }
  }
  return result;
}

// ========== Tooltip ==========
function showTooltip(event, d) {
  const tooltip = document.getElementById('tooltip');
  let html = '<strong>' + d.label + '</strong>';
  if (d.type) html += '<br>Type: ' + d.type;
  if (d.shape) html += '<br>Shape: [' + d.shape.join(', ') + ']';
  if (d.requires_grad !== undefined) html += '<br>Requires Grad: ' + d.requires_grad;
  if (d.class_name) html += '<br>Class: ' + d.class_name;
  if (d.depth !== undefined) html += '<br>Depth: ' + d.depth;
  tooltip.innerHTML = html;
  tooltip.style.left = (event.pageX + 12) + 'px';
  tooltip.style.top = (event.pageY - 10) + 'px';
  tooltip.classList.add('visible');
}
function hideTooltip() {
  document.getElementById('tooltip').classList.remove('visible');
}

// ========== Info Panel ==========
function showNodeInfo(d) {
  document.getElementById('infoTitle').textContent = d.label;
  let html = '';
  html += '<div class="info-section"><div class="info-section-title">Basic Info</div>';
  html += infoRow('Type', d.type || '-');
  if (d.subtype) html += infoRow('Subtype', d.subtype);
  if (d.class_name) html += infoRow('Class', d.class_name);
  if (d.depth !== undefined) html += infoRow('Depth', d.depth);
  html += infoRow('ID', d.id);
  html += '</div>';

  if (d.shape) {
    html += '<div class="info-section"><div class="info-section-title">Tensor Info</div>';
    html += infoRow('Shape', '[' + d.shape.join(', ') + ']');
    html += infoRow('Elements', d.shape.reduce(function(a,b){return a*b;}, 1));
    html += infoRow('Requires Grad', d.requires_grad ? 'Yes' : 'No');
    html += '</div>';
  }

  if (d.parameters && d.parameters.length > 0) {
    html += '<div class="info-section"><div class="info-section-title">Parameters</div>';
    d.parameters.forEach(function(p) {
      var totalParams = p.shape.reduce(function(a,b){return a*b;}, 1);
      html += infoRow(p.name, '[' + p.shape.join(', ') + '] (' + totalParams + ')');
    });
    html += '</div>';
  }

  if (d.children && d.children.length > 0) {
    html += '<div class="info-section"><div class="info-section-title">Children</div>';
    d.children.forEach(function(c) { html += infoRow('', c); });
    html += '</div>';
  }

  document.getElementById('infoContent').innerHTML = html;
}

function infoRow(key, value) {
  return '<div class="info-row"><span class="info-key">' + key + '</span><span class="info-value">' + value + '</span></div>';
}

function togglePanel() {
  var panel = document.getElementById('infoPanel');
  panel.classList.toggle('hidden');
  document.getElementById('statsBar').classList.toggle('panel-hidden', panel.classList.contains('hidden'));
}

function resetZoom() {
  d3.select('#graphSvg').transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity);
}

// ========== Export ==========
function exportSVG() {
  var svgEl = document.getElementById('graphSvg');
  var serializer = new XMLSerializer();
  var svgStr = '<?xml version="1.0" encoding="UTF-8"?>' + serializer.serializeToString(svgEl);
  downloadBlob(new Blob([svgStr], { type: 'image/svg+xml' }), 'tinytorch_graph.svg');
}

function exportPNG() {
  var svgEl = document.getElementById('graphSvg');
  var serializer = new XMLSerializer();
  var svgStr = serializer.serializeToString(svgEl);
  var canvas = document.createElement('canvas');
  var ctx = canvas.getContext('2d');
  var img = new Image();
  var svgBlob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  var url = URL.createObjectURL(svgBlob);
  img.onload = function() {
    canvas.width = img.width * 2; canvas.height = img.height * 2;
    ctx.scale(2, 2);
    ctx.fillStyle = '#0f1117';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    canvas.toBlob(function(blob) { downloadBlob(blob, 'tinytorch_graph.png'); }, 'image/png');
  };
  img.src = url;
}

function downloadBlob(blob, filename) {
  var url = URL.createObjectURL(blob);
  var a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

window.addEventListener('load', loadData);
window.addEventListener('resize', function() { if (graphData) renderCurrentTab(); });
</script>
</body>
</html>
"""
