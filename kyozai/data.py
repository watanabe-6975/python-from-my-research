"""教材で何度も使う計算結果（β=10 の IV マップ）を、作るか読み込むかする。
1回目は計算して data/ に .npz で保存し（約20秒）、2回目からは読むだけにする。"""
import numpy as np
from .paths import DATA
from .rcsj_mini import iv_map

BETA = 10.0
O1 = 0.5


def iv_beta10(O2=0.8, recompute=False):
    """β=10, Ω1=0.5 の IV マップ。電流は研究の保存データと同じ刻み 0.0070779 で、
    V=0.3 付近だけ（I=1.6〜4.2）を計算して時間を節約する。振幅 A は 0〜64 の 33 点。"""
    f = DATA / f"iv_beta10_O2_{O2:.2f}.npz"
    if f.exists() and not recompute:
        d = np.load(f)
        return {key: d[key] for key in d.files}
    I = np.arange(1.6, 4.2, 0.0070779)
    A = np.linspace(0, 64, 33)
    V = iv_map(I, A, BETA, O1, O2)
    np.savez(f, I=I, A=A, V=V, beta=BETA, O1=O1, O2=O2)
    return dict(I=I, A=A, V=V, beta=np.array(BETA), O1=np.array(O1), O2=np.array(O2))
