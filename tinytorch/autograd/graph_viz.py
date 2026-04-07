"""动态计算图可视化模块。

提供计算图的遍历、结构提取和可视化功能。
从输出 Tensor 出发，沿 creator 链回溯，构建完整的计算图拓扑。

Author: TinyAI Team
"""

from typing import Dict, List, Any, Optional


def extract_graph(output_tensor, max_depth: int = 100) -> Dict[str, Any]:
    """从输出 Tensor 提取计算图结构。

    从给定的输出 Tensor 开始，沿 creator 链进行 BFS 回溯，
    收集所有 Tensor 节点和 Function 操作节点，构建可视化所需的图数据。
    每个节点包含 depth 字段，用于从左到右的层级布局（depth=0 为输出层）。

    注意：必须在 backward() 之前调用（或使用 retain_graph=True），
    否则 creator 链会被 unchain_backward() 清除。

    Args:
        output_tensor: 计算图的输出 Tensor
        max_depth: 最大回溯深度，防止超大图

    Returns:
        包含 nodes 和 edges 的字典，节点含 depth 字段（0=输出，越大越靠近输入）
    """
    from tinytorch.autograd.tensor import Tensor

    nodes = []
    edges = []
    visited_tensors = set()
    visited_functions = set()
    # 记录每个节点 id 到 depth 的映射，用于后续反转
    node_depths = {}

    queue = [(output_tensor, 0)]

    while queue:
        current, depth = queue.pop(0)

        if depth > max_depth:
            continue

        if not isinstance(current, Tensor):
            continue

        tensor_id = f"tensor_{id(current)}"
        if tensor_id in visited_tensors:
            continue
        visited_tensors.add(tensor_id)

        is_leaf = current.creator is None
        node_label = current.name or "Tensor"
        shape_dims = list(current.value.shape.dims) if current.value is not None else None

        node_depths[tensor_id] = depth

        nodes.append({
            'id': tensor_id,
            'type': 'tensor',
            'subtype': 'leaf' if is_leaf else 'intermediate',
            'label': node_label,
            'shape': shape_dims,
            'requires_grad': current.requires_grad,
            'depth': depth,
        })

        if current.creator is not None:
            func = current.creator
            func_id = f"func_{id(func)}"

            if func_id not in visited_functions:
                visited_functions.add(func_id)

                func_name = func.__class__.__name__
                func_depth = depth + 1
                node_depths[func_id] = func_depth

                nodes.append({
                    'id': func_id,
                    'type': 'function',
                    'label': func_name,
                    'depth': func_depth,
                })

                for input_tensor in func.inputs:
                    input_id = f"tensor_{id(input_tensor)}"
                    edges.append({'source': input_id, 'target': func_id})
                    queue.append((input_tensor, depth + 2))

            edges.append({'source': func_id, 'target': tensor_id})

    # 反转 depth：让叶节点（输入）depth=0 在最左侧，输出 depth=max 在最右侧
    if node_depths:
        max_d = max(node_depths.values())
        for node in nodes:
            node['depth'] = max_d - node['depth']

    return {'nodes': nodes, 'edges': edges}


def extract_module_graph(module) -> Dict[str, Any]:
    """从 nn.Module 提取模块层级结构图。

    遍历 Module 的子模块树，构建层级结构的可视化数据。
    每个节点包含 depth（嵌套层级）和 order（同级顺序）字段，
    以及 parent 和 children 信息，用于层次嵌套布局。

    Args:
        module: nn.Module 实例

    Returns:
        包含 nodes 和 edges 的字典
    """
    nodes = []
    edges = []
    # 跟踪每个父节点下的子节点顺序
    parent_child_count = {}

    for name, mod in module.named_modules():
        node_id = name if name else 'root'
        class_name = mod.__class__.__name__

        # 计算嵌套深度
        depth = 0 if not name else name.count('.') + 1

        # 计算同级顺序
        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
        elif name:
            parent_name = 'root'
        else:
            parent_name = None

        order = 0
        if parent_name is not None:
            order = parent_child_count.get(parent_name, 0)
            parent_child_count[parent_name] = order + 1

        # 收集参数信息
        param_info = []
        for param_name, param in mod._parameters.items():
            shape = list(param.value.shape.dims) if param.value is not None else []
            param_info.append({'name': param_name, 'shape': shape})

        # 收集直接子模块名
        children = [
            (child_name if not name else f"{name}.{child_name}")
            for child_name in mod._modules.keys()
        ]

        nodes.append({
            'id': node_id,
            'type': 'module',
            'label': f"{class_name}" if not name else f"{name} ({class_name})",
            'class_name': class_name,
            'parameters': param_info,
            'depth': depth,
            'order': order,
            'parent': parent_name,
            'children': children,
        })

        # 添加父子边（按顺序）
        if parent_name is not None:
            edges.append({'source': parent_name, 'target': node_id})

    return {'nodes': nodes, 'edges': edges}


def visualize_graph(output_tensor, port: int = 8098, auto_open: bool = True,
                    module=None) -> None:
    """启动 Web 服务器可视化计算图。

    Args:
        output_tensor: 计算图的输出 Tensor（可为 None，仅查看模块结构时）
        port: Web 服务端口
        auto_open: 是否自动打开浏览器
        module: nn.Module 实例（可选），同时展示模块层级结构
    """
    import json
    import threading
    import webbrowser
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from tinytorch.ml.graph_template import GRAPH_HTML

    graph_data = {}
    if output_tensor is not None:
        graph_data['computation_graph'] = extract_graph(output_tensor)
    if module is not None:
        graph_data['module_graph'] = extract_module_graph(module)

    data_json = json.dumps(graph_data, ensure_ascii=False)

    class GraphHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                content = GRAPH_HTML.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            elif self.path == '/api/graph_data':
                content = data_json.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(('0.0.0.0', port), GraphHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}"
    print(f"🔍 Computation Graph Visualizer started at {url}")

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    return server


def export_graph_html(output_tensor, file_path: str, module=None) -> None:
    """将计算图导出为独立 HTML 文件（可离线查看）。

    Args:
        output_tensor: 计算图的输出 Tensor
        file_path: 输出 HTML 文件路径
        module: nn.Module 实例（可选）
    """
    import json
    from tinytorch.ml.graph_template import GRAPH_HTML

    graph_data = {}
    if output_tensor is not None:
        graph_data['computation_graph'] = extract_graph(output_tensor)
    if module is not None:
        graph_data['module_graph'] = extract_module_graph(module)

    data_json = json.dumps(graph_data, ensure_ascii=False)

    inject_script = f"""
<script>
(function() {{
  const OFFLINE_DATA = {data_json};
  window._offlineGraphData = OFFLINE_DATA;
}})();
</script>
"""
    html_content = GRAPH_HTML.replace('</body>', inject_script + '</body>')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"📄 Computation graph exported to {file_path}")
