# -*- coding: utf-8 -*-
"""Parallel segmented downloader for DIV2K (or any range-supporting URL).

Splits the file into N ranges fetched concurrently by threads, writes them into a
pre-allocated file, and verifies the final size. Resumable at chunk granularity:
already-complete chunks are skipped on rerun (sidecar .part file per chunk).
"""
import os
import sys
import threading
import time
import urllib.request

OUTDIR = os.environ.get("SR_DATASETS", "datasets")
TASKS = [
    ("https://hf-mirror.com/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip",
     "DIV2K_train.zip", 4),
]
CHUNK = 64 << 20      # 64 MB per request
lock = threading.Lock()
done_bytes = [0]


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def fetch_range(url, start, end):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 + attempt * 2)


def worker(url, path, chunks, fh, fsize):
    for (start, end) in chunks:
        part = "%s.%d.part" % (path, start)
        if os.path.exists(part) and os.path.getsize(part) == end - start + 1:
            with lock:
                done_bytes[0] += end - start + 1
            continue
        data = fetch_range(url, start, end)
        with open(part, "wb") as pf:
            pf.write(data)
        with open(path, "r+b") as f:
            f.seek(start)
            f.write(data)
        with lock:
            done_bytes[0] += len(data)
            pct = 100.0 * done_bytes[0] / fsize
            print("  %5.1f%%  %6.1f / %.1f MB" % (pct, done_bytes[0] / 1e6, fsize / 1e6),
                  flush=True)


def download(url, name, nthreads):
    path = os.path.join(OUTDIR, name)
    fsize = head_size(url)
    if fsize is None:
        sys.exit("no content-length for " + url)
    if os.path.exists(path + ".done"):
        print("%s already complete (%.1f MB)" % (name, fsize / 1e6), flush=True)
        return
    print("\n=== %s  %.2f GB  %d threads ===" % (name, fsize / 1e9, nthreads), flush=True)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.truncate(fsize)
    ranges = [(s, min(s + CHUNK - 1, fsize - 1)) for s in range(0, fsize, CHUNK)]
    with lock:
        done_bytes[0] = 0
    groups = [[] for _ in range(nthreads)]
    for i, r in enumerate(ranges):
        groups[i % nthreads].append(r)
    ths = [threading.Thread(target=worker, args=(url, path, g, None, fsize)) for g in groups if g]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    got = os.path.getsize(path)
    print("%s done: %d bytes in %.0fs  (%s)" %
          (name, got, time.time() - t0, "OK" if got == fsize else "SIZE MISMATCH"), flush=True)
    if got == fsize:
        open(path + ".done", "w").close()
        for f in os.listdir(OUTDIR):
            if f.startswith(name + ".") and f.endswith(".part"):
                os.remove(os.path.join(OUTDIR, f))


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    for url, name, th in TASKS:
        download(url, name, th)
    print("ALL DONE", flush=True)
