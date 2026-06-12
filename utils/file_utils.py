"""
知识库文件处理工具函数。

提供可测试的文件路径计算逻辑，用于将上传的文件复制到知识库目录时保留合理的目录结构。
"""
import os
import re
import shutil
from typing import List, Tuple, Optional


def _is_temp_hash_dir(dir_name: str) -> bool:
    """判断目录名是否为临时哈希目录（Gradio 上传时生成的）"""
    if not dir_name:
        return False
    return (
        bool(re.match(r"^[a-fA-F0-9]{8,}$", dir_name))
        or dir_name.lower() == "gradio"
        or dir_name.lower() == "tmp"
        or dir_name.lower() == "temp"
    )


def _get_file_stem(file_path: str) -> str:
    """获取文件名（不含扩展名）"""
    return os.path.splitext(os.path.basename(file_path))[0]


def calculate_dest_paths(
    file_paths: List[str],
    db_path: str,
) -> List[Tuple[str, str]]:
    """
    计算源文件到目标知识库目录的路径映射。

    核心逻辑：
    1. 找到所有文件的公共目录 (common_root)
    2. 若 common_root 是用户的有意义文件夹（如 算法图解、easy_rl），
       则用其父目录作为 base_root，保留最外层文件夹名
    3. 若 common_root 是 Gradio 临时哈希目录（纯 hex 名 / "gradio"），
       则直接以它为 base_root，避免把哈希名带进知识库
    4. **特殊处理**：当上传单个文件且父目录是哈希目录时，
       使用文件名（不含扩展名）作为子目录名，确保知识库有独立目录

    参数:
        file_paths: 源文件绝对路径列表
        db_path: 知识库目录路径 (如 data_base/kownledge_db)

    返回:
        [(src_path, dest_path), ...] 源文件到目标路径的映射列表
    """
    if not file_paths:
        return []

    file_paths = [os.path.abspath(p) for p in file_paths]

    # 计算公共目录
    if len(file_paths) == 1:
        common_root = os.path.dirname(file_paths[0])
    else:
        common_root = os.path.commonpath(file_paths)
        if os.path.isfile(common_root):
            common_root = os.path.dirname(common_root)

    # 判断公共目录名是否为临时哈希目录
    common_name = os.path.basename(common_root)
    is_temp_hash = _is_temp_hash_dir(common_name)

    # 确定基准目录
    if is_temp_hash:
        base_root = common_root
    else:
        base_root = os.path.dirname(common_root) or common_root

    # 计算每个文件的目标路径
    result = []
    for src in file_paths:
        rel = os.path.relpath(src, base_root)
        rel_parts = [p for p in rel.split(os.sep) if p not in ("", "..")]
        if not rel_parts:
            rel_parts = [os.path.basename(src)]
        
        # **关键修复**：当只有一个文件且父目录是哈希目录时，
        # 使用文件名（不含扩展名）作为子目录名
        # 例如：上传 "算法图解.pdf" → 保存为 "kownledge_db/算法图解/算法图解.pdf"
        if len(file_paths) == 1 and len(rel_parts) == 1 and is_temp_hash:
            file_stem = _get_file_stem(src)
            if file_stem:
                rel_parts = [file_stem, rel_parts[0]]
        
        dest = os.path.join(db_path, *rel_parts)
        result.append((src, dest))

    return result


def copy_files_to_knowledge_db(
    file_paths: List[str],
    db_path: str,
) -> List[str]:
    """
    将文件复制到知识库目录，保留合理的目录结构。

    参数:
        file_paths: 源文件路径列表
        db_path: 知识库目录路径

    返回:
        复制后的目标文件路径列表
    """
    path_mappings = calculate_dest_paths(file_paths, db_path)
    saved_files = []

    os.makedirs(db_path, exist_ok=True)

    for src, dest in path_mappings:
        os.makedirs(os.path.dirname(dest) or db_path, exist_ok=True)
        shutil.copy2(src, dest)
        saved_files.append(dest)

    return saved_files


def expand_directory_to_files(paths: List[str]) -> List[str]:
    """
    将路径列表中的目录展开为文件列表。

    参数:
        paths: 文件或目录路径列表

    返回:
        展开后的文件绝对路径列表
    """
    file_paths = []
    for p in paths:
        if os.path.isfile(p):
            file_paths.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for root, _, fnames in os.walk(p):
                for fname in fnames:
                    file_paths.append(os.path.abspath(os.path.join(root, fname)))
    return file_paths
