#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("$ ", " ".join(cmd))
    subprocess.run(cmd, check=False)


def stats(path: Path):
    lines = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = len(lines)
    lens = [len((x.get("text") or "")) for x in lines]
    avg = sum(lens) / len(lens) if lens else 0
    return {"records": n, "avg_chars": int(avg)}


def main():
    html_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pages")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("artifacts")
    out.mkdir(parents=True, exist_ok=True)
    a = out / "extract_a_clean.jsonl"
    b = out / "extract_b_bs4.jsonl"
    run(["biosyn-kb", "extract", "--html-dir", str(html_dir), "--out", str(a), "--clean"])
    run(["biosyn-kb", "extract", "--html-dir", str(html_dir), "--out", str(b)])
    sa = stats(a)
    sb = stats(b)
    print("A (clean)  :", sa)
    print("B (bs4)    :", sb)
    if sa["avg_chars"] > sb["avg_chars"]:
        print("cleaner yields longer average text (likely better de-boilerplating)")
    else:
        print("bs4 yields longer average text (cleaner might be stricter)")


if __name__ == "__main__":
    main()
