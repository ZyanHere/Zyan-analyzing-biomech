"""B4: Is the 8fps ceiling caused by auto-exposure in low light, or by the hardware?"""
import cv2, time, hashlib, numpy as np

def unique_fps(cap, secs=3.0):
    for _ in range(10): cap.read()
    last=None; uniq=0; t_end=time.perf_counter()+secs; bright=[]
    while time.perf_counter() < t_end:
        ok,f = cap.read()
        if not ok: continue
        h = hashlib.blake2b(f.tobytes(), digest_size=8).digest()
        if h != last:
            uniq += 1; last = h
            if uniq % 5 == 0: bright.append(float(f.mean()))
    return uniq/secs, (np.mean(bright) if bright else 0)

def trial(label, res, auto, exp):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    if auto is not None: cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
    if exp  is not None: cap.set(cv2.CAP_PROP_EXPOSURE, exp)
    got_auto = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE); got_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    fps, br = unique_fps(cap)
    cap.release()
    print(f"{label:<28} {res[0]}x{res[1]:<5} auto={got_auto:<5.2f} exp={got_exp:<6.1f} "
          f"-> {fps:5.1f} unique fps   mean brightness {br:5.1f}/255")
    return fps

print("baseline (whatever the camera decides on its own)")
trial("auto exposure, as-is", (640,480), None, None)
print()
print("forcing manual exposure, progressively shorter")
for e in [-4, -5, -6, -7, -8]:
    trial(f"manual exposure 2^{e}s", (640,480), 0.25, e)
print()
print("does resolution change the real rate?")
for res in [(320,240),(640,480),(1280,720)]:
    trial("manual 2^-6, varying res", res, 0.25, -6)
