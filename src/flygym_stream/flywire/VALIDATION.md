# Y-maze validation — 2026-10-14

Tested headlessly with Python 3.12, FlyGym 2.1.0 and MuJoCo 3.9.0. The package APIs were checked against the installed FlyGym source.

- Scripted left trial, default maze, seed 0, wall collisions enabled: left choice at 2.2551 simulated seconds. No escape. 114 trajectory samples. Approximately 21.9 wall seconds on the test environment; this is not a performance promise for other hardware.
- Scripted right trial: right choice at 2.2541 simulated seconds. No escape. Approximately 25.2 wall seconds while another test was also running.
- Additional constant-drive contact exercise: 20 logged samples with real fly-to-wall contacts; no escape before choice. This checks that walls are collidable, not that climbing is impossible under all policies.
- Geometry checks: start/junction inside; point beyond fork apex outside; both source positions classified in correct choice zones.
- Odor symmetry and no-odor checks passed.
- Choice dwell reset on zone exit and subsequent right choice passed.
- Neural integration scheduling: a deliberately synthetic four-node fixture substituted for the real loader exercised the BANC execution branch; 0.03 simulated seconds at 0.5 ms neural dt gave exactly 60 updates and finite completed physics. This is a software test only, not validation of the BANC network or odor-to-choice behavior.
- Python syntax checks passed. Final minor logging change ensures the last timestep is logged and precomputes the loop length.

Not tested: Windows interactive viewer, actual full BANC connection data, real-connectome behavioral response, exhaustive geometry variants, GPU acceleration. Full-brain mode needs the user's existing matching data and working load_network.py/neural_model.py. Scripted mode and maze preview need no connectome files.

The example_left_trajectory.png is the actual trace from the scripted left test, not a neural prediction.
