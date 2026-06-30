# What we are doing — simple diagram

```
 ┌──────────────────────────── 1. COLLECT DATA ────────────────────────────┐
 │  Privileged teacher drives (TM autopilot / ACC gap-keeper).             │
 │  Sensors record: RADAR distance + YOLO traffic lights.                  │
 │  Label = the ego's smoothed future speed (what a good driver did).      │
 │                                                                          │
 │   collect_throttle_brake_data.py ─► dataset_throttle_brake/data.csv      │
 │       (free driving + traffic-light + emergency + cut-in phases)         │
 │                                                                          │
 │   collect_scenario_data.py       ─► dataset_throttle_brake/data_staged.csv│
 │       (staged S1 stopped / S2 brake / S3 constant / S4 cut-in)           │
 └───────────────────────────────────┬──────────────────────────────────────┘
                                      │  every *.csv in the folder
                                      ▼
                       ┌──────────── 2. TRAIN ────────────┐
                       │   train_throttle_brake.py        │
                       │   radar+TL features ─► MLP ─►     │
                       │   target speed                   │
                       └───────────────┬──────────────────┘
                                       ▼
                            model_throttle_brake/  (the MLP)

 ┌──────────────────────────── 3. EVALUATE ────────────────────────────────┐
 │  scenarios/ : s1 stopped | s2 decelerating | s4 cut-in   (S3 off)        │
 │                                                                          │
 │   --driver mlp  ─►  radar ─► MLP ─► PID ─► throttle/brake                 │
 │   --driver pcla ─►  PCLA tfv6_visiononly (end-to-end)                     │
 │        BasicAgent steers BOTH; staging sets up the tailgate, then hands   │
 │        longitudinal control to the model at the critical moment.          │
 │                                   │                                       │
 │                                   ▼                                       │
 │            GroundTruthLogger ─► results_<driver>/*.csv                     │
 │                                   │                                       │
 │                                   ▼                                       │
 │            analyze_results.py ─► CDFs + collision rate (MLP vs PCLA)       │
 └───────────────────────────────────────────────────────────────────────────┘
```

**One line:** teach an MLP to brake/follow like a good teacher (from radar), then
race it against a big pretrained agent (PCLA) on the same staged near-crash scenarios.
