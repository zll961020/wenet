#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract WebDataset-style shards and write text/wav.scp"
    )
    p.add_argument("--shards-root", required=True,
                   help="Input root that contains subfolders like dev/test_meeting/test_net/train_l with *.tar")
    p.add_argument("--output-root", required=True,
                   help="Where to extract files and write text/wav.scp, mirrors the subsets")
    p.add_argument("--subsets", nargs="*", default=None,
                   help="Specific subsets to process (default: auto detect subfolders under shards-root)")
    p.add_argument("--wav-root", default=None,
                   help="Optional POSIX prefix to use in wav.scp paths. "
                        "If set, wav.scp uses wav-root + relative path under output-root. "
                        "Useful when训练在Linux而解压在Windows/网络盘。")
    p.add_argument("--force", action="store_true",
                   help="Recreate text/wav.scp even if they exist")
    return p.parse_args()

def posix_join(*parts):
    return str(PurePosixPath(*[str(x).replace("\\", "/") for x in parts]))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def extract_tar(tar_path: Path, dest_dir: Path):
    ensure_dir(dest_dir)
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(dest_dir)

def collect_pairs(scan_dir: Path):
    wavs = {}
    txts = {}
    for root, _, files in os.walk(scan_dir):
        r = Path(root)
        for f in files:
            if f.lower().endswith(".wav"):
                base = Path(f).stem
                wavs[base] = r / f
            elif f.lower().endswith(".txt"):
                base = Path(f).stem
                txts[base] = r / f
    # intersect keys
    keys = sorted(set(wavs.keys()) & set(txts.keys()))
    return [(k, wavs[k], txts[k]) for k in keys]

def read_text(txt_file: Path):
    try:
        s = txt_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        s = txt_file.read_text(errors="ignore")
    # 单条样本转写通常整文件就是一句；做下清洗
    s = s.strip().replace("\r\n", "\n").replace("\r", "\n")
    # 若多行，拼成一行（常见安全处理）
    if "\n" in s:
        s = " ".join([x.strip() for x in s.splitlines() if x.strip()])
    return s

def build_wav_path(wav_file: Path, output_root: Path, wav_root: str | None):
    if wav_root:
        rel = wav_file.resolve().relative_to(output_root.resolve())
        return posix_join(wav_root, rel)
    # 默认直接用POSIX风格绝对路径，避免空格/反斜杠问题
    return str(PurePosixPath(wav_file.resolve().as_posix()))

def main():
    args = parse_args()
    shards_root = Path(args.shards_root)
    output_root = Path(args.output_root)

    if not shards_root.is_dir():
        print(f"ERR: shards-root not found: {shards_root}", file=sys.stderr)
        sys.exit(1)

    # 自动发现子集目录
    subsets = args.subsets
    if not subsets:
        subsets = [p.name for p in shards_root.iterdir() if p.is_dir()]
        subsets.sort()

    for subset in subsets:
        in_dir = shards_root / subset
        if not in_dir.exists():
            print(f"SKIP: subset not found under shards-root: {subset}")
            continue

        # 输出结构：output_root/subset/{extracted/, text, wav.scp}
        out_dir = output_root / subset
        extracted_dir = out_dir / "extracted"
        ensure_dir(extracted_dir)
        ensure_dir(out_dir)

        text_path = out_dir / "text"
        wavscp_path = out_dir / "wav.scp"
        if (text_path.exists() or wavscp_path.exists()) and not args.force:
            print(f"INFO: {subset} already has text/wav.scp, use --force to overwrite.")
            # 仍然尝试增量提取（不覆盖现有清单）
        # 先清空再写
        with open(text_path, "w", encoding="utf-8") as f: pass
        with open(wavscp_path, "w", encoding="utf-8") as f: pass

        # 遍历所有 tar
        tars = sorted([p for p in in_dir.glob("*.tar") if p.is_file()])
        if not tars:
            print(f"WARN: no *.tar found in {in_dir}")
            continue

        seen = set()
        for i, tarf in enumerate(tars, 1):
            sub_dest = extracted_dir / tarf.stem  # e.g. shards_0000000000
            if not sub_dest.exists() or not any(sub_dest.iterdir()):
                print(f"[{subset}] Extracting ({i}/{len(tars)}): {tarf.name}")
                extract_tar(tarf, sub_dest)
            else:
                print(f"[{subset}] Already extracted: {tarf.name}")

            pairs = collect_pairs(sub_dest)
            if not pairs:
                print(f"WARN: no pairs found in {sub_dest}")
                continue

            with open(text_path, "a", encoding="utf-8") as f_txt, \
                 open(wavscp_path, "a", encoding="utf-8") as f_wav:
                for uttid, wavf, txtf in pairs:
                    if uttid in seen:
                        # 同一utt出现多次则跳过，避免重复
                        continue
                    seen.add(uttid)
                    txt = read_text(txtf)
                    wavp = build_wav_path(wavf, output_root, args.wav_root)
                    # 写入
                    f_txt.write(f"{uttid} {txt}\n")
                    f_wav.write(f"{uttid} {wavp}\n")

        print(f"Done subset: {subset} → {text_path} , {wavscp_path}")

if __name__ == "__main__":
    main()