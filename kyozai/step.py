"""IV 曲線からシャピロステップの幅を取り出す。
研究コード shapiro-research/py/load.py の extract（MATLAB の extract_step_width と同じ手順）と同じ。"""
import numpy as np


def extract(I, V, Vt, thr, sr, return_group=False):
    """I, V: IV 曲線 / Vt: ステップの電圧 / thr: 傾き |dV/dI| のしきい値 / sr: 電圧の窓の半幅
    戻り値: ステップ幅 ΔI（見つからなければ 0）"""
    dI = np.diff(I)
    dV = np.diff(V)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(dV / dI)
    d[~np.isfinite(d)] = np.inf
    Im = (I[1:] + I[:-1]) / 2          # 隣り合う点の中点
    Vm = (V[1:] + V[:-1]) / 2
    on = (np.abs(Vm - Vt) < sr) & (d < thr)
    if not on.any():
        return (0.0, None) if return_group else 0.0
    idx = np.where(on)[0]
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)   # 連続した塊に分ける
    best = 0.0
    bg = None
    for g in groups:
        if len(g) < 2:
            continue
        w = Im[g[-1]] - Im[g[0]]
        if w > best:
            best = w
            bg = g
    if return_group:
        return best, (None if bg is None else (Vm[bg].mean(), Im[bg[0]], Im[bg[-1]]))
    return best
