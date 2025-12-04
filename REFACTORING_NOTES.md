# 重构说明

## 已完成的重构

1. ✅ **创建统一的配置模块** (`app/core/graphrag.py`)
   - 统一导出 `GRAPHRAG_AVAILABLE`、`constructor`、`decomposer`、`retriever` 等
   - 所有服务层现在从此模块导入

2. ✅ **迁移上传功能** (`app/services/upload_service.py`)
   - 从 `backend.py` 迁移文件上传和处理逻辑
   - 支持文本、JSON、PDF/图片（通过 MinerU）处理

3. ✅ **迁移数据集管理功能** (`app/services/dataset_service.py`)
   - 数据集列表、上传 schema、删除数据集等功能
   - Schema 管理功能

4. ✅ **更新路由层**
   - 所有路由文件已更新，不再从 `backend.py` 导入
   - `upload.py`、`datasets.py`、`status.py`、`graph.py`、`qa.py` 已更新

5. ✅ **简化 backend.py**
   - `backend.py` 现在只保留启动入口
   - 从 2400+ 行减少到 11 行

## 待完成的工作

### 思维导图功能迁移 ⚠️

`app/services/mindmap_service.py` 当前为占位实现，需要从 git 历史中恢复完整功能。

**恢复步骤：**
1. 从 git 历史恢复原始 backend.py：
   ```bash
   git show HEAD:backend.py > /tmp/backend_original.py
   ```

2. 提取思维导图相关函数（约 1000+ 行代码），主要包括：
   - `find_mineru_source_file()`
   - `list_mineru_md_files()`
   - `parse_markdown_to_mindmap()`
   - `parse_content_list_to_mindmap()`
   - `generate_mindmap()`
   - `materialize_mindmap()`
   - `get_mindmap()`
   - `get_mindmap_visualization()`
   - `get_mindmap_tree()`
   - `mindmap_qa()`
   - `mindmap_qa_md()`
   - 以及大量辅助函数（`_slugify`, `_build_section_hierarchy`, 等）

3. 将这些函数迁移到 `app/services/mindmap_service.py` 或 `app/utils/mindmap_utils.py`

4. 更新导入，使用 `app.core.graphrag` 中的组件而不是从 `backend` 导入

**涉及的函数：**
- `parse_markdown_to_mindmap()`
- `parse_content_list_to_mindmap()`
- `find_mineru_source_file()`
- `list_mineru_md_files()`
- `generate_mindmap()`
- `materialize_mindmap()`
- `get_mindmap()`
- `get_mindmap_visualization()`
- `get_mindmap_tree()`
- `mindmap_qa()`
- `mindmap_qa_md()`
- 以及大量辅助函数

## 代码清理建议

1. **检查未使用的导入和函数**
2. **删除重复代码**
3. **统一错误处理**
4. **添加类型注解**

## 注意事项

- `main.py` 是 CLI 入口，应保留
- 思维导图功能代码量很大，需要仔细迁移以确保功能完整

