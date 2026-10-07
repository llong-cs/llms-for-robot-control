# Third-party notices

Original Agentic Framework and Ditto Bench code, configuration files and documentation are licensed under the [MIT License](LICENSE). Copies of this license are included in both Python packages.

Bundled third-party code retains its original licenses and copyright notices:

| Component | License | Included notice |
| --- | --- | --- |
| [Inspect Robots](https://github.com/robocurve/inspect-robots) runtime | MIT | [LICENSE](agentic-framework/vendor/inspect-robots/LICENSE) |
| Inspect Robots agent tools | MIT | [LICENSE](agentic-framework/src/agentic_framework/_vendor/inspect_robots_agent/LICENSE) |
| [OpenPi](https://github.com/Physical-Intelligence/openpi) client | Apache-2.0 | [LICENSE](agentic-framework/src/agentic_framework/_vendor/openpi_client/LICENSE) |
| [msgpack-numpy](https://github.com/lebedov/msgpack-numpy) code adapted by the OpenPi client | BSD-3-Clause | [LICENSE](agentic-framework/src/agentic_framework/_vendor/openpi_client/LICENSE.msgpack-numpy) |

The framework's Python distribution declares `MIT AND Apache-2.0 AND BSD-3-Clause` to account for these bundled components. Its original code is MIT licensed. The Ditto Bench Python distribution declares `MIT`.

Simulator/model repositories, pretrained checkpoints, datasets and simulator assets are fetched separately, at recorded revisions, by the setup and download commands. Their original licenses and access conditions apply. Assets derived from upstream resources, including the generated table meshes that reuse ManiSkill geometry and materials, retain the applicable upstream terms. The MIT license for this project's code does not relicense those resources. See the source pins in model profiles and `agentic-framework/scripts/setup/fetch_sources.py`.
