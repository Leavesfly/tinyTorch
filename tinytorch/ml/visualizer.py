"""训练可视化模块。

提供基于 Web 的训练过程可视化能力，包括：
- 实时损失曲线、准确率等指标的交互式图表
- 训练过程动画回放
- 参数拖拽调整与对比
- 图表导出为 PNG/SVG，状态保存为 JSON

使用 Python 标准库 http.server 提供 Web 服务，前端基于 Plotly.js。

Author: TinyAI Team
"""

import json
import os
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Any


class _VisualizerHandler(BaseHTTPRequestHandler):
    """可视化 Web 服务的 HTTP 请求处理器。

    处理以下请求：
    - GET /                  → 返回训练指标可视化 HTML 页面
    - GET /api/training_data → 返回训练数据 JSON
    - GET /graph             → 返回计算图可视化 HTML 页面
    - GET /api/graph_data    → 返回计算图数据 JSON
    """

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._serve_html()
        elif self.path == '/api/training_data':
            self._serve_training_data()
        elif self.path == '/graph':
            self._serve_graph_html()
        elif self.path == '/api/graph_data':
            self._serve_graph_data()
        else:
            self.send_error(404, 'Not Found')

    def _serve_html(self):
        from tinytorch.ml.viz_template import VISUALIZER_HTML
        content = VISUALIZER_HTML.encode('utf-8')
        self._send_html_response(content)

    def _serve_graph_html(self):
        from tinytorch.ml.graph_template import GRAPH_HTML
        content = GRAPH_HTML.encode('utf-8')
        self._send_html_response(content)

    def _send_html_response(self, content: bytes):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_training_data(self):
        visualizer = self.server.visualizer
        data = visualizer.get_data_snapshot()
        self._send_json_response(data)

    def _serve_graph_data(self):
        visualizer = self.server.visualizer
        data = visualizer.get_graph_snapshot()
        self._send_json_response(data)

    def _send_json_response(self, data: dict):
        content = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """静默日志，避免每次请求都打印到控制台。"""
        pass


class TrainingVisualizer:
    """训练可视化器。

    在训练过程中收集指标数据，并通过内置 Web 服务器提供交互式可视化界面。

    Attributes:
        port: Web 服务端口号
        train_losses: 每个 epoch 的训练损失列表
        val_losses: 每个 epoch 的验证损失列表
        custom_metrics: 自定义指标字典 {name: [values]}
        status: 当前训练状态 ('idle' / 'running' / 'done')

    Example:
        >>> from tinytorch.ml import TrainingVisualizer
        >>>
        >>> viz = TrainingVisualizer(port=8097)
        >>> viz.start_server()
        >>>
        >>> # 在训练循环中记录数据
        >>> for epoch in range(100):
        ...     train_loss = train_one_epoch()
        ...     viz.record_epoch(epoch, train_loss=train_loss)
        >>>
        >>> viz.finalize()
    """

    def __init__(self, port: int = 8097, auto_open: bool = True):
        """初始化可视化器。

        Args:
            port: Web 服务端口号，默认 8097
            auto_open: 启动服务后是否自动打开浏览器
        """
        self.port = port
        self.auto_open = auto_open

        # 训练数据
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.custom_metrics: Dict[str, List[float]] = {}
        self.current_epoch: int = 0
        self.start_time: Optional[float] = None
        self.status: str = 'idle'

        # 计算图数据
        self._graph_data: Dict[str, Any] = {}

        # 服务器
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start_server(self) -> None:
        """启动 Web 可视化服务器（后台线程）。"""
        self._server = HTTPServer(('0.0.0.0', self.port), _VisualizerHandler)
        self._server.visualizer = self

        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()

        url = f"http://localhost:{self.port}"
        print(f"🔥 tinyTorch Visualizer started at {url}")

        if self.auto_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    def stop_server(self) -> None:
        """停止 Web 服务器。"""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._server_thread = None
            print("🔥 tinyTorch Visualizer stopped")

    def begin_training(self) -> None:
        """标记训练开始，重置计时器。"""
        with self._lock:
            self.start_time = time.time()
            self.status = 'running'

    def record_epoch(self, epoch: int,
                     train_loss: Optional[float] = None,
                     val_loss: Optional[float] = None,
                     **extra_metrics: float) -> None:
        """记录一个 epoch 的训练指标。

        Args:
            epoch: 当前 epoch 编号（从 0 开始）
            train_loss: 训练损失
            val_loss: 验证损失（可选）
            **extra_metrics: 其他自定义指标，如 accuracy=0.95, lr=0.001
        """
        with self._lock:
            self.current_epoch = epoch + 1

            if train_loss is not None:
                self.train_losses.append(float(train_loss))

            if val_loss is not None:
                self.val_losses.append(float(val_loss))

            for metric_name, metric_value in extra_metrics.items():
                if metric_name not in self.custom_metrics:
                    self.custom_metrics[metric_name] = []
                self.custom_metrics[metric_name].append(float(metric_value))

    def finalize(self) -> None:
        """标记训练完成。"""
        with self._lock:
            self.status = 'done'
        print(f"✅ Training completed. Visualizer still running at http://localhost:{self.port}")

    def set_graph(self, output_tensor=None, module=None) -> None:
        """设置要可视化的计算图和/或模块结构。

        设置后可通过 http://localhost:{port}/graph 查看交互式计算图。

        注意：必须在 backward() 之前调用（或使用 retain_graph=True），
        否则 creator 链会被清除。

        Args:
            output_tensor: 计算图的输出 Tensor，用于提取动态计算图
            module: nn.Module 实例，用于提取模块层级结构
        """
        from tinytorch.autograd.graph_viz import extract_graph, extract_module_graph

        with self._lock:
            self._graph_data = {}
            if output_tensor is not None:
                self._graph_data['computation_graph'] = extract_graph(output_tensor)
            if module is not None:
                self._graph_data['module_graph'] = extract_module_graph(module)

        graph_url = f"http://localhost:{self.port}/graph"
        print(f"🔍 Computation graph available at {graph_url}")

    def get_graph_snapshot(self) -> Dict[str, Any]:
        """获取当前计算图数据的快照（线程安全）。

        Returns:
            包含计算图和模块结构的字典
        """
        with self._lock:
            return dict(self._graph_data)

    def get_data_snapshot(self) -> Dict[str, Any]:
        """获取当前训练数据的快照（线程安全）。

        Returns:
            包含所有训练指标的字典
        """
        with self._lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            return {
                'train_losses': list(self.train_losses),
                'val_losses': list(self.val_losses),
                'custom_metrics': {k: list(v) for k, v in self.custom_metrics.items()},
                'current_epoch': self.current_epoch,
                'elapsed_time': round(elapsed, 2),
                'status': self.status,
            }

    def save_data(self, file_path: str) -> None:
        """将训练数据保存为 JSON 文件。

        Args:
            file_path: 保存路径（.json）
        """
        snapshot = self.get_data_snapshot()
        snapshot['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        print(f"📊 Training data saved to {file_path}")

    def load_data(self, file_path: str) -> None:
        """从 JSON 文件加载训练数据。

        Args:
            file_path: JSON 文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with self._lock:
            self.train_losses = data.get('train_losses', [])
            self.val_losses = data.get('val_losses', [])
            self.custom_metrics = data.get('custom_metrics', {})
            self.current_epoch = data.get('current_epoch', 0)
            self.status = data.get('status', 'done')

        print(f"📊 Training data loaded from {file_path}")

    def export_html(self, file_path: str) -> None:
        """将当前可视化导出为独立 HTML 文件（含数据，可离线查看）。

        Args:
            file_path: 输出 HTML 文件路径
        """
        from tinytorch.ml.viz_template import VISUALIZER_HTML

        snapshot = self.get_data_snapshot()
        data_json = json.dumps(snapshot, ensure_ascii=False)

        # 将内嵌数据注入 HTML，替换 fetch 逻辑为直接加载
        inject_script = f"""
<script>
// Injected offline data
(function() {{
  const OFFLINE_DATA = {data_json};
  const originalRefresh = window.refreshData || function(){{}};
  window.refreshData = function() {{
    trainingData = OFFLINE_DATA;
    updateAll();
    setStatus(trainingData.status || 'done');
    if (window.autoRefreshTimer) {{ clearInterval(window.autoRefreshTimer); }}
  }};
}})();
</script>
"""
        html_content = VISUALIZER_HTML.replace('</body>', inject_script + '</body>')

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 Visualization exported to {file_path}")

    def __repr__(self) -> str:
        return (f"TrainingVisualizer(port={self.port}, "
                f"epochs={self.current_epoch}, status='{self.status}')")

    def __del__(self):
        """析构时自动停止服务器。"""
        self.stop_server()
