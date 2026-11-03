"""ファイルの場所は、この教材リポジトリの直下（このファイルの1つ上のフォルダ）を基準に決める。
研究コードの py/paths.py と同じ考え方。どこから実行しても同じ場所を指す。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # 教材リポジトリの直下
DATA = ROOT / "data"                             # 教材で作った計算結果（.npz）
DATA.mkdir(exist_ok=True)
