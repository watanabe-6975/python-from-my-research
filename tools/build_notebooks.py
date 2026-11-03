"""tools/src/*.txt（章の原稿）から notebooks/*.ipynb を作る。

原稿の書き方：
  ##MD   … ここからマークダウンのセル
  ##CODE … ここからコードのセル
各章の最初のマークダウンセルの直後に、教材フォルダを import できるようにするセルを自動で入れる。
使い方：python tools/build_notebooks.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "tools" / "src"
OUT = ROOT / "notebooks"

SETUP = '''# 準備：教材の部品（kyozai/）を import できるようにする。どの章でも最初に1回だけ実行する。
import sys
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "kyozai").exists():   # kyozai/ があるフォルダまで上にたどる
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
print("教材の場所:", ROOT.name)'''


def parse(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.strip() in ("##MD", "##CODE"):
            if kind:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("md" if line.strip() == "##MD" else "code"), []
        else:
            buf.append(line)
    if kind:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def build(src):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    first_md_done = False
    for kind, body in parse(src.read_text(encoding="utf-8")):
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(body))
            if not first_md_done:
                nb.cells.append(nbf.v4.new_code_cell(SETUP))
                first_md_done = True
        else:
            nb.cells.append(nbf.v4.new_code_cell(body))
    out = OUT / (src.stem + ".ipynb")
    nbf.write(nb, out)
    print("作成:", out.relative_to(ROOT))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for src in sorted(SRC.glob("*.txt")):
        build(src)
