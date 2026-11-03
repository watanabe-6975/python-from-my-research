"""二色 RCSJ モデル  γ'' + β γ' + sin γ = I + A (sin Ω1 τ + sin Ω2 τ)  を RK4 で解く。

研究コード shapiro-research/py/rcsj.py（MATLAB の rk4_steady と同じ手順）を、
変数名と説明を足して書き直したもの。計算の手順は同じ：
  - 電流の点ごとに γ=0, γ'=initial, τ=0 から始める
  - N_run ステップ捨ててから、N_avg ステップの γ' を平均して V とする
  - V が負なら 0 にする
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def accel(t, g, G, I, b, A, O1, O2):
    """γ'' を返す（運動方程式を γ'' について解いた右辺）。g=γ, G=γ'"""
    return I - b * G - np.sin(g) + A * np.sin(O1 * t) + A * np.sin(O2 * t)


@njit(cache=True)
def steady_voltage(I, b, A, O1, O2, dt, N_run, N_avg, initial):
    """1つの電流 I について、平均電圧 V = <γ'> を返す。"""
    g = 0.0
    G = initial
    t = 0.0
    s = 0.0
    for k in range(N_run + N_avg):
        k1 = accel(t, g, G, I, b, A, O1, O2) * dt;                   j1 = G * dt
        k2 = accel(t + dt/2, g + j1/2, G + k1/2, I, b, A, O1, O2) * dt; j2 = (G + k1/2) * dt
        k3 = accel(t + dt/2, g + j2/2, G + k2/2, I, b, A, O1, O2) * dt; j3 = (G + k2/2) * dt
        k4 = accel(t + dt, g + j3, G + k3, I, b, A, O1, O2) * dt;       j4 = (G + k3) * dt
        g += (j1 + 2*j2 + 2*j3 + j4) / 6
        G += (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
        if k >= N_run:
            s += G
    v = s / N_avg
    return v if v > 0 else 0.0


@njit(parallel=True, cache=True)
def iv_map(Iarr, Alist, b, O1, O2, dt=1/32, N_run=20106, N_avg=20106, initial=1.0):
    """電流 Iarr × 振幅 Alist のすべての組で V を求める。戻り値の形は (電流の数, 振幅の数)。"""
    M = Iarr.size
    nA = Alist.size
    out = np.zeros((M, nA))
    for p in prange(M * nA):          # prange: CPU のコアに仕事を分ける
        i = p // nA
        a = p % nA
        out[i, a] = steady_voltage(Iarr[i], b, Alist[a], O1, O2, dt, N_run, N_avg, initial)
    return out


def iv_numpy(Iarr, b, A, O1, O2, dt=1/32, N_run=20106, N_avg=20106, initial=1.0):
    """NumPy 版：電流の方向をまとめて（ベクトル化して）同時に進める。
    MATLAB の pf_rk4_iv.m と同じ書き方。numba がなくても動く。"""
    I = np.asarray(Iarr, dtype=float)
    g = np.zeros_like(I)
    G = np.full_like(I, initial)
    t = 0.0
    S = np.zeros_like(I)

    def f(gg, GG, tt):
        return I - b * GG - np.sin(gg) + A * np.sin(O1 * tt) + A * np.sin(O2 * tt)

    for k in range(N_run + N_avg):
        k1g = G;              k1G = f(g, G, t)
        k2g = G + dt/2 * k1G; k2G = f(g + dt/2 * k1g, G + dt/2 * k1G, t + dt/2)
        k3g = G + dt/2 * k2G; k3G = f(g + dt/2 * k2g, G + dt/2 * k2G, t + dt/2)
        k4g = G + dt * k3G;   k4G = f(g + dt * k3g, G + dt * k3G, t + dt)
        g = g + dt/6 * (k1g + 2*k2g + 2*k3g + k4g)
        G = G + dt/6 * (k1G + 2*k2G + 2*k3G + k4G)
        t += dt
        if k >= N_run:
            S += G
    return np.maximum(S / N_avg, 0.0)
