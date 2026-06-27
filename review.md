# Scenario Visual Guide — What to Watch For

Run: `python3 run_all.py` (or `python3 s1_lead_vehicle_stopped.py` for a single scenario)

---

## S1 — Lead Vehicle Stopped (NHTSA #25)

**What happens:**
1. Your ego car (Tesla Model 3, autopilot) spawns on a straight highway
2. It starts driving forward at ~0-40 km/h
3. At **step 100** (~5s in), a stopped NPC vehicle spawns **35 metres ahead**
4. The ego car should detect it and stop before hitting it

**What to watch in CARLA:**
- 👉 **Camera:** Chase cam shows your car from behind + above
- 👉 Look **ahead** — you should see a parked car 35m in front
- 👉 Watch if your car slows down and stops (distance should close to ~5-8m)
- 👉 If you see the parked car and your car stops without hitting it → ✅

**What gets logged:** collision (yes/no), minimum distance reached, ego speed at each step

**Expected:** Autopilot stops at ~5-8m from obstacle at all fog levels

---

## S2 — Lead Vehicle Decelerating (NHTSA #4)

**What happens:**
1. Your ego car spawns on a straight highway
2. An NPC car spawns **25 metres ahead**, driving at 30 km/h
3. Both drive forward — ego tries to follow at ~35 km/h
4. At **step 300** (~15s in), the NPC **slams its brakes** and stops suddenly
5. The ego should react and brake to avoid rear-ending the NPC

**What to watch in CARLA:**
- 👉 **Camera:** Chase cam shows your car + the NPC ahead
- 👉 At first: both cars driving, distance stays ~25m
- 👉 After NPC brakes: watch if your car slows down
- 👉 If your car stops without hitting the NPC → ✅

**What gets logged:** collision, minimum distance, reaction time (steps from NPC brake to ego brake)

**Expected:** Autopilot should brake and stop, but in practice it often doesn't react well (the ego may keep going at constant speed even after the NPC stops)

---

## S3 — Lead Vehicle Constant Speed (NHTSA #12)

**What happens:**
1. Your ego car spawns targeting 40 km/h
2. An NPC car spawns **35 metres ahead**, driving at a constant **20 km/h**
3. Since ego is faster (40 km/h) than NPC (20 km/h), the gap closes over time
4. The ego should detect the slower NPC and match its speed

**What to watch in CARLA:**
- 👉 **Camera:** Chase cam behind your car
- 👉 At first: gap closes as ego is faster
- 👉 Watch if your car slows down to match the NPC's speed
- 👉 If distance stabilises (doesn't keep getting smaller) → ✅
- 👉 If your car rear-ends the NPC → ❌

**What gets logged:** collision, minimum distance, following distance over time, speed difference

**Expected:** Autopilot is mixed here — sometimes it matches speed, sometimes it doesn't notice the slower NPC

---

## Fog Modes

| Fog | Visibility | What you'll see |
|-----|-----------|-----------------|
| 0 (light) | Clear, can see far ahead | Normal driving |
| 100 (heavy) | Very limited, ~20m | Barely see the car in front |

---

## Quick Visual Checklist

When watching a run, check these in order:

| Step | What to see | S1 | S2 | S3 |
|------|------------|----|----|----|
| Start | Car on highway, camera behind | ✅ | ✅ | ✅ |
| 5-10s | NPC appears ahead | ✅ Parked car | ✅ Moving NPC | ✅ Moving NPC |
| Mid-run | Ego approaching NPC | ✅ Closing in | ✅ Following | ✅ Gap closing |
| After NPC brake | Ego slows down | ✅ Should stop | ✅ Should brake | ✅ Should match speed |
| End | Ego stopped safely | ✅ ~5-8m gap | ✅ No crash | ✅ ~10-15m gap |

If at any point you see the car going **through** another vehicle → **collision detected** (logged as 💥)
