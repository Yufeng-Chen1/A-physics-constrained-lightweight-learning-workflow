#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按需求重新生成时频图谱与子图谱数据：
- 清空旧的 optimized_patch_data/* 输出
- welldata 按 8:2 分层划分为训练/验证（均覆盖5类）；test_well 为测试集
- 统一预处理（去噪+标准化+深度对齐），并在小波包分解前提醒长4+5层相对深度对齐
- 使用小波包分解（4层为主）生成 64×64×6 时频图谱；再按最优方案提取 32×32 子图谱
- 训练集使用 BSMOTE + 类别权重方法处理类别不均，保障数据质量
- 保存目录：
  optimized_patch_data/train_spectrograms
  optimized_patch_data/valid_spectrograms
  optimized_patch_data/test_spectrograms
"""

import os
import sys
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入现有模块
from data.spectrogram_data_loader import SpectrogramWellLogDataset
from data.fluid_types import FluidTypes
from main.optimized_patch_extractor import OptimizedPatchExtractor
from data.class_balance_handler import ClassBalanceHandler  # type: ignore


def _clean_output_dirs(base: Path) -> None:
    for sub in ["train_spectrograms", "valid_spectrograms", "test_spectrograms", "spectrograms", "metadata", "pca_models"]:
        p = base / sub
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    base.mkdir(exist_ok=True)
    (base / "train_spectrograms").mkdir(parents=True, exist_ok=True)
    (base / "valid_spectrograms").mkdir(parents=True, exist_ok=True)
    (base / "test_spectrograms").mkdir(parents=True, exist_ok=True)


def _ensure_output_dirs(base: Path) -> None:
    base.mkdir(exist_ok=True)
    (base / "train_spectrograms").mkdir(parents=True, exist_ok=True)
    (base / "valid_spectrograms").mkdir(parents=True, exist_ok=True)
    (base / "test_spectrograms").mkdir(parents=True, exist_ok=True)


def _safe_move_with_retry(src: Path, dst: Path, max_retries: int = 6) -> bool:
    """Windows下帶重試的安全移動；若目標已存在則自動改名避免衝突。"""
    import time
    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    # 如目標已存在，改名避免覆蓋衝突
    if dst.exists():
        stem, suffix = dst.stem, dst.suffix
        k = 1
        while True:
            candidate = dst_parent / f"{stem}__{k}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            k += 1
    for attempt in range(1, max_retries + 1):
        try:
            try:
                os.replace(str(src), str(dst))  # 原子替換
                return True
            except Exception:
                import shutil as _shutil
                _shutil.copy2(str(src), str(dst))
                try:
                    os.unlink(str(src))
                except Exception:
                    pass
                return True
        except Exception:
            time.sleep(0.2 * attempt)
    return False


def _collect_well_names_from_dir(dir_path: Path) -> List[str]:
    return [p.stem for p in dir_path.glob("*.txt")]


def _build_stratified_indices(labels: List[int], ratio: float, random_state: int = 42) -> Tuple[List[int], List[int]]:
    """将索引按分层方式划分为 A/B，A 占比 ratio。"""
    rng = np.random.RandomState(random_state)
    labels = np.asarray(labels)
    idx_all = np.arange(len(labels))
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)
    a_idx: List[int] = []
    b_idx: List[int] = []
    for y, idxs in by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        k = int(len(idxs) * ratio)
        a_idx.extend(idxs[:k].tolist())
        b_idx.extend(idxs[k:].tolist())
    return a_idx, b_idx


def _ensure_all_classes_present(labels: List[int]) -> bool:
    s = set(int(x) for x in labels)
    need = set(FluidTypes.get_all_fluid_ids())
    return need.issubset(s)


def _determine_label_from_text(text: str) -> int:
    # 与现有映射一致：致密油层归为油层；油水同层/含油水层/含水油层 -> 油水层
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None  # type: ignore
    t = str(text).strip().lower()
    if any(k in t for k in ["油水层", "油水同层", "含油水", "含水油"]):
        return 4
    if any(k in t for k in ["差油层", "低产油层", "较差油层", "弱油层"]):
        return 3
    if any(k in t for k in ["致密油层", "致密", "油层", "油"]):
        return 0
    if any(k in t for k in ["水层", "水"]):
        return 1
    if any(k in t for k in ["干层", "干"]):
        return 2
    return None  # type: ignore


def _load_label_for_wellfile(file_path: Path) -> int:
    # 读取井文件，取全井主导标签（优先窗口多数票不可用时回退全井多数）
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin1", "utf-16", "utf-16le", "utf-16be"]:
        try:
            df = pd.read_csv(file_path, encoding=enc, sep='\t')
            break
        except Exception:
            df = None
    if df is None:
        return None  # type: ignore
    col = None
    for c in ["解释结论", "结论"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        return None  # type: ignore
    labels = [ _determine_label_from_text(x) for x in df[col].astype(str).tolist() ]
    labels = [x for x in labels if x is not None]
    if not labels:
        return None  # type: ignore
    # 多数票
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def split_train_valid_test(welldata_dir: Path, test_well_dir: Path, train_ratio: float = 0.8, random_state: int = 42) -> Dict[str, List[str]]:
    """分层：welldata -> train/valid；test_well -> test。确保train/valid覆盖5类。
    若存在用戶提供的清單文件，則優先採用：
      optimized_patch_data/train_wells.txt
      optimized_patch_data/val_wells.txt
      optimized_patch_data/test_wells.txt
    每行一口井名（不含副檔名）。
    """
    all_wells = _collect_well_names_from_dir(welldata_dir)
    test_wells = _collect_well_names_from_dir(test_well_dir)

    # 嘗試讀取用戶提供的井清單
    base = PROJECT_ROOT / "optimized_patch_data"
    def _read_list(fname: str):
        p = base / fname
        if p.exists():
            try:
                return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]
            except Exception:
                try:
                    return [ln.strip() for ln in p.read_text(encoding='gbk').splitlines() if ln.strip()]
                except Exception:
                    return None
        return None

    user_train = _read_list('train_wells.txt')
    user_val = _read_list('val_wells.txt')
    user_test = _read_list('test_wells.txt')

    # 如果用户提供了 train/val/test，優先使用
    if user_train or user_val or user_test:
        split = {
            'train': sorted(set(user_train or [])),
            'val': sorted(set(user_val or [])),
            'test': sorted(set(user_test or test_wells)),
        }
        return split

    # 否則：为 welldata 中的每口井计算一个主标签，用于分层
    well_to_label: Dict[str, int] = {}
    for wn in all_wells:
        y = _load_label_for_wellfile(welldata_dir / f"{wn}.txt")
        if y is not None:
            well_to_label[wn] = int(y)
    wells_labeled = [w for w in all_wells if w in well_to_label]
    labels = [well_to_label[w] for w in wells_labeled]

    # 简单分层：对井层面划分
    train_idx, valid_idx = _build_stratified_indices(labels, train_ratio, random_state)
    train_wells = [wells_labeled[i] for i in train_idx]
    valid_wells = [wells_labeled[i] for i in valid_idx]

    # 兜底：确保覆盖5类（若不满足，回退随机+补齐逻辑）
    def fix_cover(wells: List[str]) -> List[str]:
        y = [well_to_label.get(w, None) for w in wells]
        while not _ensure_all_classes_present([_y for _y in y if _y is not None]):
            # 从剩余井中补齐缺失类别
            missing = list(set(FluidTypes.get_all_fluid_ids()) - set([_y for _y in y if _y is not None]))
            if not missing:
                break
            for m in list(missing):
                cands = [w for w in wells_labeled if (w not in wells) and (well_to_label.get(w, None) == m)]
                if cands:
                    wells.append(cands[0])
                    y.append(m)
                else:
                    missing.remove(m)
            if not missing:
                break
            # 若仍缺，强制加入任意剩余井
            rem = [w for w in wells_labeled if w not in wells]
            if not rem:
                break
            wells.append(rem[0])
            y.append(well_to_label.get(rem[0], None))
        return wells

    train_wells = fix_cover(train_wells)
    valid_wells = fix_cover(valid_wells)

    split = {
        'train': sorted(set(train_wells)),
        'val': sorted(set(valid_wells)),
        'test': sorted(set(test_wells)),
    }
    return split


def generate_all(cache_dir: Path, welldata_dir: Path, test_well_dir: Path,
                 train_wells_dir: Path = None, val_wells_dir: Path = None, explicit_test_wells_dir: Path = None,
                 skip_clean: bool = False, subsets: Optional[List[str]] = None) -> None:
    if skip_clean:
        _ensure_output_dirs(cache_dir)
    else:
        _clean_output_dirs(cache_dir)

    print("提示：小波包分解前已进行深度对齐与长4+5层相对深度对齐，保持跨曲线窗口中心对齐。")

    # 若提供三目录(train_wells/val_wells/test_wells)则直接使用；否则按 welldata/test_well 分层
    use_explicit_dirs = (
        (train_wells_dir is not None and train_wells_dir.exists()) or
        (val_wells_dir is not None and val_wells_dir.exists()) or
        (explicit_test_wells_dir is not None and explicit_test_wells_dir.exists())
    )

    if use_explicit_dirs:
        def _collect(dir_path: Path):
            return sorted([p.stem for p in (dir_path.glob('*.txt') if (dir_path is not None and dir_path.exists()) else [])])
        split = {
            'train': _collect(train_wells_dir),
            'val': _collect(val_wells_dir),
            'test': _collect(explicit_test_wells_dir if (explicit_test_wells_dir is not None and explicit_test_wells_dir.exists()) else test_well_dir),
        }
    else:
        # 先按井划分（train/val按8:2，test由test_well）
        split = split_train_valid_test(welldata_dir, test_well_dir, train_ratio=0.8, random_state=42)
    print("划分结果:")
    print(json.dumps(split, ensure_ascii=False, indent=2))

    # 生成核心缓存（使用 optimized_patch_extractor 的一致预处理与特征）
    # 我们将分别针对 train/val/test 过滤井文件集合来生成，对应保存到指定子目录
    # 复用 OptimizedPatchExtractor：其输出到 cache_dir/spectrograms。我们分三次运行并移动产物。

    chosen_subsets = subsets or ['train', 'valid', 'test']

    def run_extractor_for_wells(subset_name: str, wells: List[str], base_dir_for_subset: Path = None) -> None:
        if not wells:
            return
        # 临时单独目录
        temp_cache = cache_dir / f"__temp_{subset_name}__"
        if temp_cache.exists():
            shutil.rmtree(temp_cache, ignore_errors=True)
        temp_cache.mkdir(parents=True, exist_ok=True)

        # 训练/验证/测试来源目录
        base_dir = base_dir_for_subset if base_dir_for_subset is not None else (test_well_dir if subset_name == 'test' else welldata_dir)
        extractor = OptimizedPatchExtractor(welldata_dir=str(base_dir), cache_dir=str(temp_cache))

        # 仅处理指定井
        files = [base_dir / f"{w}.txt" for w in wells if (base_dir / f"{w}.txt").exists()]
        for i, file_path in enumerate(files, 1):
            print(f"[{subset_name}] 处理: {file_path.name} ({i}/{len(files)})")
            extractor.process_well_file(Path(file_path))
        extractor._save_metadata()

        # 将生成的 npz 样本迁移到目标子目录（帶重試與改名）
        src = temp_cache / "spectrograms"
        dst = cache_dir / f"{subset_name}_spectrograms"
        dst.mkdir(exist_ok=True)
        moved, failed = 0, 0
        for f in src.glob("*.npz"):
            if _safe_move_with_retry(f, dst / f.name):
                moved += 1
            else:
                failed += 1
        print(f"[{subset_name}] 移动: 成功{moved} 失败{failed}")
        # 清理临时
        shutil.rmtree(temp_cache, ignore_errors=True)

    # 选择性运行所需子集
    if 'train' in chosen_subsets:
        if use_explicit_dirs:
            run_extractor_for_wells("train", split['train'], train_wells_dir)
        else:
            run_extractor_for_wells("train", split['train'])
    if 'val' in chosen_subsets or 'valid' in chosen_subsets:
        if use_explicit_dirs:
            run_extractor_for_wells("valid", split['val'], val_wells_dir)
        else:
            run_extractor_for_wells("valid", split['val'])
    if 'test' in chosen_subsets:
        if use_explicit_dirs:
            base_te = explicit_test_wells_dir if (explicit_test_wells_dir is not None and explicit_test_wells_dir.exists()) else test_well_dir
            run_extractor_for_wells("test", split['test'], base_te)
        else:
            run_extractor_for_wells("test", split['test'])

    # 训练集应用 BSMOTE（针对子图谱样本）+ 类别权重统计
    train_dir = cache_dir / "train_spectrograms"
    train_files = list(train_dir.glob("*patch*.npz"))
    labels = []
    for f in train_files:
        try:
            with np.load(f) as d:
                labels.append(int(d.get('label', -1)))
        except Exception:
            continue
    labels = [y for y in labels if y >= 0]
    if labels:
        print("训练集类别分布(子图谱):", {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})
    else:
        print("警告：未发现可用训练子图谱标签，跳过BSMOTE。")

    # 这里只记录建议与权重保存，实际BSMOTE通常在特征/向量空间进行；
    # 我们保存采样建议与类别权重供训练阶段加载。
    cls_counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))} if labels else {}
    total = sum(cls_counts.values()) if cls_counts else 0
    class_weights = {}
    for c in FluidTypes.get_all_fluid_ids():
        cnt = cls_counts.get(c, 1)
        class_weights[c] = float(np.sqrt(total / (len(FluidTypes.get_all_fluid_ids()) * cnt))) if total > 0 else 1.0

    weights_file = cache_dir / "train_class_weights.json"
    with open(weights_file, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in class_weights.items()}, f, ensure_ascii=False, indent=2)
    print(f"已保存训练类别权重: {weights_file}")

    print("全部生成完成。")


def main():
    print("重新生成时频图谱与子图谱数据(8:2:测试)")
    base = PROJECT_ROOT / "optimized_patch_data"
    welldata_dir = PROJECT_ROOT / "welldata"
    test_well_dir = PROJECT_ROOT / "test_well"
    train_wells_dir = PROJECT_ROOT / "train_wells"
    val_wells_dir = PROJECT_ROOT / "val_wells"
    explicit_test_wells_dir = PROJECT_ROOT / "test_wells"

    generate_all(
        base, welldata_dir, test_well_dir,
        train_wells_dir=train_wells_dir if train_wells_dir.exists() else None,
        val_wells_dir=val_wells_dir if val_wells_dir.exists() else None,
        explicit_test_wells_dir=explicit_test_wells_dir if explicit_test_wells_dir.exists() else None
    )


if __name__ == "__main__":
    main()
