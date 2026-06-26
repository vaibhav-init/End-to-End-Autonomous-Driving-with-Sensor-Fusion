# Method: Vision vs Radar Comparison in Fog

## The Hypothesis

YOLO-based vision is degraded by heavy fog — the camera can't see through it,
so distance estimates are garbage or missing entirely. Radar cuts through fog
because it uses radio waves.

## The Test (scenario1.py)

Run `python3 scenario1.py` twice:

### Run 1: Camera-only model (`--mode vision`)
- Extreme fog (density=100, distance=10m)
- Ego drives at 30 km/h using PID + vision ML model
- Stopped car spawns 45m ahead
- **Expected:** YOLO can't see through fog, model doesn't brake → 💥 COLLISION

### Run 2: Radar model (`--mode radar`)
- Same extreme fog
- Same 30 km/h, same obstacle
- **Expected:** Radar sees through fog, model brakes in time → ✅ NO COLLISION

## What This Proves

| | Camera-only (YOLO) | Camera + Radar |
|---|---|---|
| **Clear weather** | Works (barely, with hack) | Works |
| **Heavy fog** | Fails — YOLO blind | Works — radar sees through |
| **Open road cruising** | Needs rule hack (30 km/h floor) | Model works naturally |
| **Obstacle braking** | Hoppy/jerky | Smoother |

## The Real Takeaway

Vision-only is a research novelty. It fails in the first bad weather condition
you throw at it. Radar is cheap (~$50), works in all weather, and makes the
control problem dramatically easier because distance is a direct measurement
instead of a noisy geometric estimate.

If this project were production, the answer would be:
**Add a radar sensor. Vision is a backup at best.**
