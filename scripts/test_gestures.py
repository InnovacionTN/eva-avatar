"""Verify the gesture system: print each gesture's peak contribution and
render baseline vs tilt/lean/shrug frames for visual inspection."""
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import eva_platica as ep  # noqa: E402

src = os.path.join(ep.PROJECT_ROOT, "assets", "eva_body.png")
r = ep.EvaRenderer(src, pasteback=True, lip_gain=1.25, body_motion=True, body_sway=1.6)

print("\n[test] peak contribution of each gesture (p=0.5):")
for name, dur in ep.GESTURES.items():
    r._gesture, r._gesture_t0 = name, time.time() - 0.5 * dur
    c = r._gesture_contrib(time.time())
    print(f"  {name:6s}: dpitch={c['dpitch']:+5.1f} dyaw={c['dyaw']:+5.1f} "
          f"droll={c['droll']:+5.1f} dscale={c['dscale']:+.3f} shrug={c['shrug']:.2f}")
r._gesture = None

out_dir = os.path.join(ep.PROJECT_ROOT, "out")
os.makedirs(out_dir, exist_ok=True)


def render_at_peak(name, dur):
    if name:
        r._gesture, r._gesture_t0 = name, time.time() - 0.5 * dur
    else:
        r._gesture = None
    return r.frame(0.0, speaking=False)


for tag, name, dur in (("baseline", None, 0), ("tilt", "tilt", 1.6),
                       ("lean", "lean", 1.6), ("shrug", "shrug", 1.0)):
    im = render_at_peak(name, dur)
    p = os.path.join(out_dir, f"gesture_{tag}.png")
    cv2.imwrite(p, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    print(f"[test] saved {p}")
