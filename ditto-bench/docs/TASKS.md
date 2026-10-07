# Task definitions

Each task has `easy`, `medium`, `hard`, and `xhard` variants. Variants retain the same instruction and success goal while changing the physical manipulation problem. IDs below are zero-based and match `environment.task_ids` in evaluation configurations.

| ID | Task | Slug | Environment ID |
| --- | --- | --- | --- |
| 0 | Constrained extraction | `slanted_board` | `DittoSlantedBoard-v1` |
| 1 | Odd-object grasping | `odd_geometry` | `DittoOddGeometry-v1` |
| 2 | Precision insertion | `precision_insert` | `DittoPrecisionInsert-v1` |
| 3 | Standing stability | `stand_object` | `DittoStandObject-v1` |
| 4 | Tool-assisted drawer | `tool_drawer` | `DittoToolDrawer-v1` |
| 5 | Ring release | `unlock_ring` | `DittoUnlockRing-v1` |

## Difficulty design

| ID | Easy | Medium | Hard | Extra hard (`xhard`) |
| --- | --- | --- | --- | --- |
| 0 | Horizontal, loose channel | Upward-sloping channel | Channel with pitch, yaw, and roll | Same tilted channel, tighter fit |
| 1 | Small, less slippery sphere | Tapered block with two less slippery faces | Slippery football shape | Slippery polyhedron with a triangular protrusion |
| 2 | Round peg, roomy hole | Round peg, tighter hole | Rectangular peg and slot | Asymmetric L-shaped peg and slot |
| 3 | Wide base, centered upper body | Narrow base, centered upper body | Narrow base, rectangular upper body | Narrow base, offset upper body |
| 4 | Rod, lighter drawer | Rod, heavier resisted drawer | Same drawer with overhead obstruction | Wide board tool, obstruction retained |
| 5 | Square ring, raised aligned side opening | Lower opening requiring lifting | Upward opening requiring rotation and lifting | Larger oval ring, upward opening |

## Success and progress

| ID | Success | Progress |
| --- | --- | --- |
| 0 | Entire board clears the channel. | Extraction distance along the channel from reset. |
| 1 | Complete horizontal footprint inside the basket; object released. | Bottom-of-object lift toward basket-rim clearance. |
| 2 | Valid, fully seated insertion; released and still for one second. | Valid insertion depth. |
| 3 | Gray base down on the table, upper body clear, no robot contact, and stable rest for three seconds. | Upright orientation improvement plus continuous standing time. |
| 4 | Drawer at its opening target, with enough tool-attributed opening, no direct-hand violation, and settled motion. | Net tool-attributed opening; closing subtracts credit. |
| 5 | Whole ring and its enclosed interior clear the bridge; holding is allowed. | Reduction in ring-mouth distance to an available bridge-side exit. |

Task 1 requires no basket-floor support or sustained holding; protruding above the rim is allowed. Lifting alone can reach full progress without placement. For task 4, hand-driven opening invalidates tool use; stationary cabinet contact does not. Task 5 can reach full progress before complete removal.

[Execution contracts](API.md) · [Task implementation](../src/ditto/tasks)
