# Task definitions

Benchmark version: `0.7.4`. Scoring version: `physical-relative-progress-v7`.

Task IDs are zero-based. Each task has `easy`, `medium`, `hard`, and `xhard` variants.

| ID | Task | Slug | Environment ID |
| --- | --- | --- | --- |
| 0 | Constrained extraction | `slanted_board` | `DittoSlantedBoard-v1` |
| 1 | Odd-object grasping | `odd_geometry` | `DittoOddGeometry-v1` |
| 2 | Precision insertion | `precision_insert` | `DittoPrecisionInsert-v1` |
| 3 | Standing stability | `stand_object` | `DittoStandObject-v1` |
| 4 | Tool-assisted drawer | `tool_drawer` | `DittoToolDrawer-v1` |
| 5 | Ring release | `unlock_ring` | `DittoUnlockRing-v1` |

Instructions describe goals without providing oracle measurements or prescribed grasp/motion plans. Any physically effective solution that satisfies the task predicate is accepted.

## Geometry and difficulty

| ID | Easy | Medium | Hard | Extra hard |
| --- | --- | --- | --- | --- |
| 0 | Horizontal channel | Channel pitched 28 degrees | Pitched channel with 30-degree yaw and roll | Same orientation with narrower bore |
| 1 | Small sphere | Tapered block with parallel grasp faces | Rounded pointed football | Soccer-ball polyhedron with a triangular ear |
| 2 | Circular peg, 5 mm clearance | Circular peg, 2 mm clearance | Rectangular peg | L-shaped peg |
| 3 | Wide base and square upper body | Narrow base and square upper body | Narrow base and rectangular upper body | Narrow base and offset rectangular upper body |
| 4 | Rod and lighter drawer | Rod and heavier, more resistant drawer | Same drawer with an overhead eave | Same obstruction with a wide thin board tool |
| 5 | Square ring with an aligned side mouth | Centered mouth needing elevation | Square ring with its mouth initially on top | Larger open elliptical ring needing rotation/elevation |

The blue extraction board is 340 x 60 x 16 mm, initially inserted 205 mm. Channel mouths are 105 mm above the table. Ordinary bores are 66 x 22 mm; `xhard` uses 62 x 18 mm. Inclined channel assemblies use a real finite-depth table pocket.

The odd objects share a 300 g mass. Basket inner footprint is 240 x 200 mm; floor and rim are 10 and 80 mm above the table. The sphere uses friction 0.10. The medium block's two parallel end faces use 0.10, with 0.05 elsewhere; hard/xhard objects use 0.05. No software grasp gate is imposed.

Insertion sockets have 55 mm usable depth and a real matching table aperture. Standing objects share a 151 mm height and 200 g mass. Their gray disk base is 64 mm in diameter for easy and 32 mm for the other variants. Extra hard shifts the upper body center by 8 mm.

Tools start upright in 110 mm passive holders, exposing about 82 mm for grasping. Rods are 190 mm long and 12 mm in diameter; the board is 190 x 44 x 6 mm. The common bridge handle has 80 mm clear span, 33 mm clear depth, and 8 mm bars. The drawer has 145 mm travel. The hard/xhard eave underside is 136 mm above the table.

The ring bridge beam is 80 x 320 x 16 mm at a 124 mm center height. Square ring outer/inner envelopes are 168 x 168 / 140 x 140 mm. The extra-hard ellipse starts 240 mm wide and 200 mm high, with a 212 x 172 mm inner envelope. Every ring is 120 g and 28 mm thick with a real 32 mm mouth. The bridge and ring share a 45-degree nominal yaw. The geometry modules and reset asset specifications are the authoritative source for complete dimensions and materials.

## Success and progress

| ID | Success predicate | Progress measure before common reset normalization |
| --- | --- | --- |
| 0 | Entire board clears the channel mouth by the authored extraction margin, on the first eligible control step. | Rear-edge extraction distance from its reset position divided by required travel. |
| 1 | Complete object XY projection lies inside the basket and the robot has released it, on the first eligible control step. | Lowest collision-point clearance above the table, zero during tabletop contact, normalized by the 85 mm lift target. |
| 2 | Physically valid insertion, bottom seated within the socket floor tolerance, released and still for one second. | Valid insertion depth divided by 55 mm socket depth. |
| 3 | Gray base down, object touching table, upper body clear of table, robot not touching, and object at rest continuously for three seconds. | 50% upright orientation improvement from reset + 50% continuous standing time normalized by three seconds. |
| 4 | Drawer open at least 105 mm, at least 60 mm net tool-attributed opening, no direct-hand violation, speed below 40 mm/s for 0.5 seconds. | Net tool-attributed opening divided by 105 mm. Closing subtracts contribution. |
| 5 | The whole ring and its filled interior clear every bridge solid by at least 2 mm, on the first eligible control step. | Reduction from reset of the mouth center's shortest 3D distance to either accessible bridge-side exit centerline. |

For task 1, lift history, basket-floor support, velocity, and sustained holding are diagnostics rather than success requirements. The object may protrude above the rim. Progress can reach one without basket placement.

For task 3, the gray-base-down tilt tolerance is 10 degrees. The upper body must remain more than 4 mm clear of the tabletop. Contact-force threshold is 0.01 N; rest thresholds are 15 mm/s linear speed and 0.16 rad/s angular speed. Continuous standing is measured at the physics frequency: 451 consecutive samples at 150 Hz cover three seconds. Failed conditions reset the current standing run.

For task 4, a gripper-driven net opening above 2 mm permanently marks a direct hand violation. Other robot parts directly moving the drawer cannot replace tool use. Incidental contact with stationary cabinet structure does not alone establish hand opening.

For task 5, the distance metric includes horizontal, vertical, and out-of-span displacement in the live bridge frame. Reaching an exit may give progress one before complete removal; moving farther away afterwards may reduce the final score. Holding the ring is allowed.

Each task can also fail when a designated dynamic prop falls below the physical workspace. Dwell counters and peak scores advance only through actual simulator steps. See [API.md](API.md) for termination, common `done` verification, and snapshot requirements.
