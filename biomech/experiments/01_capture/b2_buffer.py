"""B2: Is OpenCV buffering frames behind our back? Buffered frame == stale frame."""
import cv2, time

def run(bufsize, secs=4.0):
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if bufsize is not None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, bufsize)
    actual_buf = cap.get(cv2.CAP_PROP_BUFFERSIZE)
    for _ in range(15):
        cap.read()
    gaps = []
    t_end = time.perf_counter() + secs
    prev = time.perf_counter()
    n = 0
    while time.perf_counter() < t_end:
        ok, f = cap.read()
        now = time.perf_counter()
        if ok:
            gaps.append((now - prev) * 1000)
            n += 1
        prev = now
    cap.release()
    gaps.sort()
    q = lambda p: gaps[min(int(len(gaps)*p), len(gaps)-1)]
    fps = n / secs
    burstiness = sum(1 for g in gaps if g < 5.0) / len(gaps) * 100
    return dict(req=bufsize, got=actual_buf, fps=fps, p05=q(0.05), p50=q(0.50),
                p95=q(0.95), max=gaps[-1], instant_pct=burstiness)

print(f"{'bufsize':>8} {'reported':>9} {'fps':>6} {'p05':>7} {'p50':>7} {'p95':>7} {'max':>7}  {'<5ms':>6}")
for b in [None, 1, 2]:
    r = run(b)
    print(f"{str(r['req']):>8} {r['got']:>9.0f} {r['fps']:>6.1f} {r['p05']:>7.1f} {r['p50']:>7.1f} "
          f"{r['p95']:>7.1f} {r['max']:>7.1f}  {r['instant_pct']:>5.0f}%")
print()
print("'<5ms' = share of reads that returned instantly, i.e. straight from a buffer.")
print("A well-behaved live source should sit near 33ms every time, with almost none <5ms.")
