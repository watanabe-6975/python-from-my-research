"""ステップ幅の理論曲線とフィット。研究コード shapiro-research/py/coherent.py と同じ式。"""
import numpy as np
from scipy.special import jv


def single(n, m, A, k, O1=0.5, O2=0.8):
    """単項：|J_n(kA/Ω1) J_m(kA/Ω2)|"""
    return np.abs(jv(n, k * A / O1) * jv(m, k * A / O2))


def coherent(n, m, A, k, O1=0.5, O2=0.8, P=8, Q=5, J=4):
    """コヒーレント和：同じ電圧に届く (n+P j, m−Q j) を位相 (−i)^(n+m) つきで足す。
    Ω1:Ω2 = 5:8 のとき 8Ω1 = 5Ω2 なので P=8, Q=5。"""
    S = np.zeros_like(np.asarray(A, dtype=float), dtype=complex)
    for j in range(-J, J + 1):
        nn, mm = n + P * j, m - Q * j
        S += (-1j) ** (nn + mm) * jv(nn, k * A / O1) * jv(mm, k * A / O2)
    return np.abs(S)


def fit(sw, A, model, n, m, ks=None, **kw):
    """k を格子状に動かし、相関係数が最大になる k を探す。
    戻り値: (相関 r, k*, 倍率 sc)。sc は最小二乗で決める。"""
    if ks is None:
        ks = np.logspace(np.log10(0.0005), np.log10(1.3), 600)
    best = (-1.0, None, None)
    for k in ks:
        bt = model(n, m, A, k, **kw)
        if bt.std() == 0 or sw.std() == 0:
            continue
        sc = (sw * bt).sum() / (bt ** 2).sum()
        r = np.corrcoef(sw, sc * bt)[0, 1]
        if r > best[0]:
            best = (r, k, sc)
    return best
