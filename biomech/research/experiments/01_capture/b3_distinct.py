"""B3: Of the frames we read, how many are actually NEW? Reads != frames."""
import cv2, time, hashlib

def run(backend_name, backend, secs=4.0, drain=False):
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):
        cap.read()
    reads = 0
    hashes = []
    last = None
    new_gaps = []
    t_prev_new = time.perf_counter()
    t_end = time.perf_counter() + secs
    while time.perf_counter() < t_end:
        ok, f = cap.read()
        if not ok:
            continue
        reads += 1
        h = hashlib.blake2b(f.tobytes(), digest_size=8).digest()
        if h != last:
            now = time.perf_counter()
            new_gaps.append((now - t_prev_new) * 1000)
            t_prev_new = now
            last = h
            hashes.append(h)
    cap.release()
    uniq = len(set(hashes))
    new_gaps = sorted(new_gaps[1:])
    q = lambda p: new_gaps[min(int(len(new_gaps)*p), len(new_gaps)-1)] if new_gaps else 0
    return dict(be=backend_name, reads=reads/secs, uniq=uniq/secs,
                dup_pct=100*(1 - uniq/max(reads,1)), p50=q(.5), p95=q(.95))

print(f"{'backend':<8} {'reads/s':>8} {'UNIQUE/s':>9} {'dupes':>7} {'new-frame gap p50':>18} {'p95':>7}")
for name, be in [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]:
    r = run(name, be)
    print(f"{r['be']:<8} {r['reads']:>8.1f} {r['uniq']:>9.1f} {r['dup_pct']:>6.0f}% "
          f"{r['p50']:>18.1f} {r['p95']:>7.1f}")
