import os
import glob
import pandas as pd

ENCODINGS = ['utf-8','utf-8-sig','gbk','gb2312','latin1','utf-16','utf-16le','utf-16be']

ALIAS_MAP = {
    '井名': {'井名','井号','井號','well_name','well','����','���'},
    '解释结论': {'解釋結論','解释結论','結論','结论','解释','說明結論','說明','���ͽ���'},
    '层位': {'層位','地層','层位名称','層位名稱','��λ'},
    '深度': {'深度','DEPTH','Depth'},
    'GR': {'GR'}, 'SP': {'SP'}, 'AC': {'AC','DT','SON'},
    'DEN': {'DEN','RHOB'}, 'CNL': {'CNL','NEU','NPHI'},
    'RT': {'RT','RES','RILD','RLLD','RT90','RT60','RT30','RT20','RT10','RT06','RT06%1','RT90%1','RILM','RLLS'}
}

REV = {}
for std, als in ALIAS_MAP.items():
    for a in als:
        REV[str(a).lower()] = std

def _load_best_df(path: str):
    """尝试多种编码+分隔方案，返回列数最多且包含关键信息的DataFrame。"""
    best = (None, None, 0, False)
    for enc in ENCODINGS:
        # 尝试制表符
        for sep, engine in (('\t', None), (r'\s+', 'python')):
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep, engine=engine)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            cols = list(df.columns)
            score = len(cols)
            has_labelish = any(str(c).lower() in REV for c in cols)
            # 优先包含潜在别名的方案
            better = (score > best[2]) or (score == best[2] and has_labelish and not best[3])
            if better:
                best = (df, enc, score, has_labelish)
    return best[0], best[1]


def normalize_file(path: str) -> tuple:
    df, used_enc = _load_best_df(path)
    if df is None or df.empty:
        return (path, 'read_failed', None)

    # normalize columns
    new_cols = []
    for c in df.columns:
        key = str(c).strip()
        lk = key.lower()
        mapped = REV.get(lk, key)
        new_cols.append(mapped)
    df.columns = new_cols

    # merge helper
    def merge_first(df: pd.DataFrame, std: str, cands: set):
        if std in df.columns:
            return
        for cand in cands:
            if cand in df.columns:
                df[std] = df[cand]
                return

    # 井名/深度占位修复（若缺失则按列位映射前两列）
    try:
        if '井名' not in df.columns and len(df.columns) >= 1:
            df.rename(columns={df.columns[0]: '井名'}, inplace=True)
        if '深度' not in df.columns and len(df.columns) >= 2:
            df.rename(columns={df.columns[1]: '深度'}, inplace=True)
    except Exception:
        pass

    # label/layer
    merge_first(df, '解释结论', ALIAS_MAP['解释结论'])
    merge_first(df, '层位', ALIAS_MAP['层位'])

    # RT preference; 苏257 -> RT90
    base = os.path.basename(path)
    if 'RT' not in df.columns:
        for c in ALIAS_MAP['RT']:
            if c in df.columns:
                df['RT'] = df[c]
                break
    if base.startswith('苏257') or base.lower().startswith('su257'):
        if 'RT90' in df.columns:
            df['RT'] = df['RT90']

    # write back utf-8
    try:
        df.to_csv(path, index=False, sep='\t', encoding='utf-8')
        return (path, used_enc, list(df.columns))
    except Exception:
        return (path, 'write_failed', None)


def main():
    roots = ['welldata', 'test_well']
    changed = []
    failed = []
    for folder in roots:
        if not os.path.isdir(folder):
            continue
        for path in glob.glob(os.path.join(folder, '*.txt')):
            res = normalize_file(path)
            if res[1] in ('read_failed','write_failed'):
                failed.append(res)
            else:
                changed.append(res)
    print('CHANGED:', len(changed))
    for p, enc, cols in changed[:20]:
        print('  ->', p, enc, 'cols=', cols[:12])
    print('FAILED:', len(failed))
    for it in failed:
        print('  !!', it)


if __name__ == '__main__':
    main()
