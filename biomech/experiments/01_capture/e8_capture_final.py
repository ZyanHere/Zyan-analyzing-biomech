"""E8 (item 10): close out the capture question with exposure fixed.

F5 measured MSMF duplicate frames BEFORE the exposure fix, so the 74% figure
may no longer hold. Also establishes the true capture latency ceiling.
"""
import cv2, time, hashlib, numpy as np

def probe(name, backend, fix_exposure, secs=5.0):
    cap = cv2.VideoCapture(0, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if fix_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    for _ in range(15): cap.read()
    reads = 0; last = None; uniq = 0; gaps = []; t_prev = time.perf_counter()
    t_end = time.perf_counter() + secs
    while time.perf_counter() < t_end:
        ok, f = cap.read()
        if not ok: continue
        reads += 1
        h = hashlib.blake2b(f.tobytes(), digest_size=8).digest()
        if h != last:
            now = time.perf_counter(); gaps.append((now-t_prev)*1000); t_prev = now
            last = h; uniq += 1
    cap.release()
    g = sorted(gaps[1:])
    q = lambda p: g[min(int(len(g)*p), len(g)-1)] if g else float('nan')
    return dict(name=name, exp="fixed" if fix_exposure else "auto",
                reads=reads/secs, uniq=uniq/secs, dup=100*(1-uniq/max(reads,1)),
                p50=q(.5), p95=q(.95), pmax=g[-1] if g else float('nan'))

print("="*80)
print("Item 10: duplicate frames and capture timing, with exposure fixed")
print("="*80)
print(f"  {'backend':<8}{'exposure':<9}{'reads/s':>9}{'UNIQUE/s':>10}{'dupes':>8}"
      f"{'gap p50':>9}{'p95':>8}{'max':>8}")
for name, be in [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]:
    for fix in (False, True):
        r = probe(name, be, fix)
        print(f"  {r['name']:<8}{r['exp']:<9}{r['reads']:>9.1f}{r['uniq']:>10.1f}"
              f"{r['dup']:>7.0f}%{r['p50']:>9.1f}{r['p95']:>8.1f}{r['pmax']:>8.1f}")
print()
print("  UNIQUE/s is the real frame rate. reads/s counts the loop, not the camera.")
print("  gap = interval between genuinely new frames, in ms.")
