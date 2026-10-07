# Procedural assets and project resources

Task props are generated from the included Python geometry modules on first environment creation. There are no manually downloaded benchmark meshes or generated binary assets in this source tree. Concave structures use separate convex components so slots, baskets, holders, and ring mouths remain physically open.

| Resource | Location relative to the project root | Override |
| --- | --- | --- |
| Generated task geometry | `data/generated/ditto-bench/v1` | `DITTO_ASSET_DIR` |
| Pinned official robot/task source | `third_party/molmoact2` | `MOLMOACT2_SOURCE_ROOT` |
| Pinned FR3/Robotiq asset snapshot | `data/molmoact2-sim-eval-assets` | `DROID_ASSET_ROOT` |
| Official ManiSkill/YCB inventory | `data/maniskill` | `MS_ASSET_DIR` (direct simulator use) |
| Preview and evaluation outputs | `outputs/` | `--output-dir` |

Framework evaluation also accepts `--molmoact2-source` and `--maniskill-assets`; those arguments are passed into the isolated simulator worker. Direct Gym use and preview read the environment variables above.

The required source commit is `66b87e64efd99dfd103241418113955cf64dfa9c` from `allenai/molmoact2`. The robot assets are `TreeePlanter/molmoact2-sim-eval-assets` at `9332a64224ff0a813d9f77bd377b845270232513`. Use the included framework asset downloader to preserve the required `VLA_SOURCE.json` inventory. See the [installation instructions](../README.md).

The shared framework simulator also verifies the official YCB inventory at startup. Run the included `download_droid_resources.py ycb` command even though the six tasks generate their own props. The benchmark's framework worker uses `data/maniskill` for this common resource root.

Dimensions are metres, rotations are WXYZ quaternions, and primitive cylinder/capsule axes are local Z. Box dimensions in specifications are half extents; cylinder/capsule lengths are half lengths. Materials declare density in kg/m³, static/dynamic friction, and zero restitution. Native capsules are rotated from SAPIEN's X-axis convention to the same local-Z geometry.

Each generated actor directory contains `spec.json`, a colored `visual.glb`, and per-component collision OBJ files with SHA-256 values. Running collisions use native boxes/spheres/capsules and convex cylinder/polyhedron meshes. Cache identity includes geometry parameters, generator source SHA-256/version, and trimesh version. Actor identity includes poses, materials, visibility, and collision flags. Source or parameter changes create a new cache entry.

Preview manifests record rendered task definitions and assets. Framework reset metadata inventories the generated files and actual primitive meshes loaded by SAPIEN, alongside source fingerprints and sampled fixture poses. Keep these manifests with results. Do not edit cached meshes manually; update geometry source and regenerate instead.

The benchmark reproduces the official camera definitions, robot base/table placement, native controllers, and 150/30 Hz settings. Task 0's finite-depth pocket and task 2's matching through-aperture are necessary physical differences from the reference table. They preserve its external texture and general placement while providing collision geometry suited to the task.

The table and robot remain part of the ManiSkill and official MolmoAct2 asset distributions. Their licenses are not replaced by this benchmark; retain the upstream notices and licenses when redistributing those resources or assets derived from them. Original benchmark code and geometry generators are covered by the [MIT License](../LICENSE). See the project's [third-party notices](../../THIRD_PARTY_NOTICES.md) for resource licensing.
