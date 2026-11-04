#!/usr/bin/env python3
"""A reduced-depth copy of a GLM-5.3-Flash (glm5_next) checkpoint, tensors byte for byte.

Keeps the embeddings, the final norm, the LM head, the first N decoder layers and the MTP
block (renamed from layer `num_hidden_layers` to layer N, where a model of N layers keeps it),
and rewrites config.json / hf_quant_config.json for N layers. Quantized tensors (NVFP4 weights,
their scales) are copied as shipped; nothing is converted.

The source may be on another host: the tensors are read there by a small stdlib-only reader
run over ssh and streamed into the new shards here, so only the kept bytes cross the link and
only they are written.

    truncate.py --src DIR [--host HOST] --dst DIR [--layers N] [--shard-gb 4]

Needs Python 3.8+ on both hosts, nothing else.
"""

import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys

# Runs on the source host: `list` prints the kept tensors (name, dtype, shape, bytes) as JSON;
# `data` streams their bytes in the same order.
READER = r'''
import json, os, re, struct, sys
mode, src, keep = sys.argv[1], sys.argv[2], set(int(x) for x in sys.argv[3].split(","))
idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
def wanted(n):
    if "visual" in n:
        return False
    m = re.match(r"model\.(?:language_model\.)?layers\.(\d+)\.", n)
    return int(m.group(1)) in keep if m else True
shards = {}
for n, f in idx.items():
    if wanted(n):
        shards.setdefault(f, []).append(n)
out, listing = sys.stdout.buffer, []
for f in sorted(shards):
    with open(os.path.join(src, f), "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
        base = 8 + hl
        for n in sorted(shards[f], key=lambda n: hdr[n]["data_offsets"][0]):
            t = hdr[n]
            a, b = t["data_offsets"]
            if mode == "list":
                listing.append([n, t["dtype"], t["shape"], b - a])
                continue
            fh.seek(base + a)
            left = b - a
            while left:
                chunk = fh.read(min(left, 1 << 24))
                if not chunk:
                    raise SystemExit("short read in " + f)
                out.write(chunk)
                left -= len(chunk)
if mode == "list":
    out.write(json.dumps(listing).encode())
'''

SIDE_FILES = [
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def remote(host, mode, src, keep):
    code = 'import base64;exec(base64.b64decode("%s"))' % base64.b64encode(READER.encode()).decode()
    args = [sys.executable if host is None else "python3", "-c", code, mode, src, keep]
    if host is not None:
        args = ["ssh", host, " ".join("'%s'" % a for a in args)]
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def read_file(host, path):
    if host is None:
        with open(path, "rb") as f:
            return f.read()
    return subprocess.run(["ssh", host, "cat '%s'" % path], check=True, stdout=subprocess.PIPE).stdout


def rename(name, mtp_src, n):
    return re.sub(r"(model\.(?:language_model\.)?layers\.)%d\." % mtp_src, r"\g<1>%d." % n, name)


def truncate_config(cfg, n):
    tc = cfg["text_config"] if "text_config" in cfg else cfg
    full = tc["num_hidden_layers"]
    tc["num_hidden_layers"] = n
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        if key in tc:
            tc[key] = tc[key][:n]
    lin = tc.get("linear_attn_config")
    if lin:
        for key in ("full_attn_layers", "kda_layers"):
            if key in lin:
                lin[key] = [i for i in lin[key] if i < n]
    if "quantization_config" in cfg:
        cfg["quantization_config"]["ignore"] = fix_modules(cfg["quantization_config"].get("ignore", []), full, n)
    return full


def fix_modules(mods, full, n):
    out = []
    for m in mods:
        g = re.match(r"(model\.(?:language_model\.)?layers\.)(\d+)(.*)", m)
        if g is None:
            out.append(m)
        elif int(g.group(2)) == full:
            out.append("%s%d%s" % (g.group(1), n, g.group(3)))
        elif int(g.group(2)) < n:
            out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="checkpoint directory (on --host)")
    ap.add_argument("--host", help="ssh host the source is on (default: this one)")
    ap.add_argument("--dst", required=True, help="new checkpoint directory")
    ap.add_argument("--layers", type=int, default=4, help="decoder layers kept (default 4)")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args()

    cfg = json.loads(read_file(a.host, os.path.join(a.src, "config.json")))
    full = truncate_config(cfg, a.layers)
    tc = cfg.get("text_config", cfg)
    n_mtp = tc.get("num_nextn_predict_layers", 0) or 0
    keep = list(range(a.layers)) + list(range(full, full + n_mtp))

    lister = remote(a.host, "list", a.src, ",".join(map(str, keep)))
    listing = json.loads(lister.stdout.read())
    if lister.wait() != 0:
        raise SystemExit("listing failed")

    # Plan the shards: tensors in stream order, a new shard past --shard-gb.
    limit = int(a.shard_gb * (1 << 30))
    shards, cur, size = [], [], 0
    for t in listing:
        if cur and size + t[3] > limit:
            shards.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += t[3]
    if cur:
        shards.append(cur)

    os.makedirs(a.dst, exist_ok=True)
    names = ["model-%05d-of-%05d.safetensors" % (i + 1, len(shards)) for i in range(len(shards))]
    weight_map, total = {}, 0
    data = remote(a.host, "data", a.src, ",".join(map(str, keep)))
    for fname, tensors in zip(names, shards):
        hdr, off = {}, 0
        for name, dtype, shape, nbytes in tensors:
            new = rename(name, full, a.layers)
            hdr[new] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
            weight_map[new] = fname
            off += nbytes
        h = json.dumps(hdr).encode()
        h += b" " * (-len(h) % 8)
        with open(os.path.join(a.dst, fname), "wb") as out:
            out.write(struct.pack("<Q", len(h)) + h)
            left = off
            while left:
                chunk = data.stdout.read(min(left, 1 << 24))
                if not chunk:
                    raise SystemExit("stream ended early")
                out.write(chunk)
                left -= len(chunk)
        total += off
        print("%s: %d tensors, %.2f GB" % (fname, len(tensors), off / 1e9), flush=True)
    if data.stdout.read(1) or data.wait() != 0:
        raise SystemExit("stream did not end where expected")

    with open(os.path.join(a.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
    with open(os.path.join(a.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    try:
        q = json.loads(read_file(a.host, os.path.join(a.src, "hf_quant_config.json")))
        qq = q.get("quantization", {})
        qq["exclude_modules"] = fix_modules(qq.get("exclude_modules", []), full, a.layers)
        with open(os.path.join(a.dst, "hf_quant_config.json"), "w") as f:
            json.dump(q, f, indent=2)
    except subprocess.CalledProcessError:
        pass
    for side in SIDE_FILES:
        try:
            body = read_file(a.host, os.path.join(a.src, side))
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        with open(os.path.join(a.dst, side), "wb") as f:
            f.write(body)
    print("%d layers + %d MTP: %d tensors, %.2f GB in %s" % (a.layers, n_mtp, len(weight_map), total / 1e9, a.dst))


if __name__ == "__main__":
    main()
