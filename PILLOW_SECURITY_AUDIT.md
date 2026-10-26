# Pillow security remediation and environment migration

Audit date: 2026-10-26. Main revision: `455c5d078c240d81fb11902c0c47c3e7aef9facc`.

This change pins Pillow to 13.0.0, the first official fixed release for the
reported October advisories. Because it requires Python >=3.11, new installations
use Python 3.11 and the matching CARLA 0.9.16 client. Existing experiment
environments must be preserved separately. This is a dependency remediation;
closed-loop numerical equivalence with Python 3.8 / CARLA 0.9.13 is not established.
Dependabot must rescan the updated main branch to confirm alert closure; no alerts
are manually dismissed by this remediation.

## Alert inventory and official fixes

Seven GitHub notification emails were verified for FREA on October 24–25 (Asia/Shanghai).
All name `requirements.txt`. Private Dependabot API state is unavailable through
the connected tools, so notification evidence is not a claim that these are all
currently open alerts. The public advisory database was also queried through OSV:
Pillow 13.0.0 returned no known vulnerability records on the audit date. The old
10.3.0 query returned additional advisories beyond the seven notifications.

| Alert | CVE | GitHub reviewed affected range | First fix | Trigger / impact |
| --- | --- | --- | --- | --- |
| [168](https://github.com/CurryChen77/FREA/security/dependabot/168) | [CVE-2026-112403](https://github.com/python-pillow/Pillow/security/advisories/GHSA-8g5g-8vmr-9782) | <=12.3.0 | 13.0.0 | Lossless SOF3 JPEG with draft/thumbnail scaling and libjpeg-turbo >=3: native heap overflow |
| [169](https://github.com/CurryChen77/FREA/security/dependabot/169) | [CVE-2026-112404](https://github.com/python-pillow/Pillow/security/advisories/GHSA-c6hw-5grf-ff96) | >=8.1.0,<13.0.0 | 13.0.0 | TIFF tile/RGBA decode row-size mismatch: out-of-bounds reads, heap disclosure or crash |
| [170](https://github.com/CurryChen77/FREA/security/dependabot/170) | [CVE-2026-112397](https://github.com/python-pillow/Pillow/security/advisories/GHSA-58r4-6rv3-gfh8) | <=12.3.0 | 13.0.0 | Zero-width image passed to native morphology: out-of-bounds access/crash |
| [171](https://github.com/CurryChen77/FREA/security/dependabot/171) | [CVE-2026-112896](https://github.com/python-pillow/Pillow/security/advisories/GHSA-pchr-8j38-3x4p) | <=12.3.0 | 13.0.0 | Saving TIFF with retained ColorMap incompatible with sample depth: heap out-of-bounds read |
| [173](https://github.com/CurryChen77/FREA/security/dependabot/173) | [CVE-2026-112400](https://github.com/python-pillow/Pillow/security/advisories/GHSA-65qm-jvmh-6pqg) | >=9.2.0,<13.0.0 | 13.0.0 | Reusing GIF after catching decompression-bomb error: rejected frame size persists, causing memory exhaustion |
| [174](https://github.com/CurryChen77/FREA/security/dependabot/174) | [CVE-2026-112895](https://github.com/python-pillow/Pillow/security/advisories/GHSA-p4wg-p73j-c528) | >=5.1.0,<13.0.0 | 13.0.0 | PDF append parses deep/cyclic page trees or long trailer chains: stack/CPU/memory exhaustion |
| [175](https://github.com/CurryChen77/FREA/security/dependabot/175) | [CVE-2026-112897](https://github.com/python-pillow/Pillow/security/advisories/GHSA-q45j-58j2-45rg) | >=5.3.0,<13.0.0 | 13.0.0 | Short JPEG/fax TIFF decode: uninitialized buffer tail returned as pixels; strip path since 6.0.0, tile path since 5.3.0 |

The upstream repository pages sometimes use broader affected ranges and lower
severity than GitHub's reviewed advisory database. In particular, email #168 says
Critical, while its upstream report says High; #170 and #173 emails say High,
while upstream reports say Low. The differences do not change the fixed version.
The table uses reviewed version ranges where they are more precise.

Official release and interpreter metadata:

- [Pillow 13 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
- [Pillow 13 PyPI metadata](https://pypi.org/pypi/pillow/13.0.0/json): `Requires-Python >=3.11`
- [Pillow Python support](https://pillow.readthedocs.io/en/stable/installation/python-support.html)

Pillow 10.4 still supports Python 3.8 but does not resolve these advisories.
Pillow 11/12 also remain affected. An old-Python fallback or ignoring
Requires-Python would leave the actual runtime vulnerable and is not used here.

## Smallest supported dependency change

- `Pillow==10.3.0` -> `Pillow==13.0.0`.
- `scikit_image==0.19.3` -> `scikit_image==0.20.0`: the old pin has no CPython
  3.11 wheel. The candidate uses the next release with a matching wheel, avoiding
  a resource-intensive unsupported source build. Existing resize calls are covered
  by legacy/candidate comparisons.
- `setup.py` Python range: `>=3.8,<3.9` -> `>=3.11,<3.12`. The narrow range reflects
  the environment actually validated, rather than promising untested interpreters.
- README: install a fresh environment and matching CARLA 0.9.16 cp311 wheel.
  No simulator binaries, drivers, system CUDA or shared environments are changed.
  Four actor autopilot control-transfer calls now pass the configured Traffic
  Manager port, avoiding an implicit fallback to shared port 8000.

The closed-loop entrypoint also exposed an undeclared dependency: `CarlaRunner`
imports TensorBoard unconditionally. `tensorboard==2.14.0` is now explicit, matching
the historical local snapshot; its OSV query returned no known records. All other
direct requirement pins are retained, including NumPy 1.24.4, SciPy
1.10.1, matplotlib 3.6.3, moviepy 1.0.3, torch 2.4.1 and Lightning 2.4.0.
The server's `requirements-all.txt` is an old untracked local snapshot, not a
manifest on GitHub main; it is not modified or presented as a supported install.
The validation environment uses the CPU build of the same torch version to avoid
CUDA downloads and GPU allocation. Its complete resolved package list is retained
with the task evidence; this is not a general upgrade of transitive dependencies
in an existing experiment environment.

## Code reachability and compatibility

Four tracked Python files import PIL:

- `frea/agent/agent_utils/viz_tokens_bev.py`: L/RGB images, polygon/line drawing,
  NumPy conversion and PNG output; imported by the PlanT dataset.
- `frea/agent/agent_utils/explainability_utils.py`: BGRA/RGB conversion, RGBA masks
  and alpha paste; called by the CARLA environment.
- `frea/agent/PlanT/PlanT_agent.py`: BEV drawing, ImageOps borders and PNG output.
- `frea/agent/expert/nav_planner.py`: route debug drawing.

No direct production `Image.open`, `draft`, `thumbnail`, TIFF decode or PDF append
call was found in tracked Python source. This limits demonstrated direct exposure,
not the vulnerable dependency's impact through extensions, tools or indirect
image decoding. Used APIs do not rely on removed Image.ANTIALIAS, font-size APIs,
or saving 32-bit I-mode PNGs. The repository imports moviepy's FFmpeg writer,
not its old Pillow resize helper (which references removed Image.ANTIALIAS).
Unrelated moviepy editor/resize workflows are not claimed compatible.

## Regression validation

`tests/pillow_regression.py` reuses PR #15's 14 decoded-array fingerprints. It
executes the actual helper bodies via AST, avoiding simulator/torch imports;
these are not full-module or closed-loop tests. All 14 passed in the legacy
Python 3.8.20 environment and the Python 3.11.14/Pillow 10.3.0 reference.
The Python 3.11.14/Pillow 13.0.0 candidate also passed all 14 with identical hashes.
Compressed PNG bytes are not
compared because encoder versions may legitimately change them.

`tests/resize_regression.py` records six arrays with Python 3.8.20, scikit-image
0.19.3, Pillow 10.3.0, NumPy 1.24.4, SciPy 1.10.1 and pygame 2.3.0. It covers the
actual display conversion helpers and the camera observation resize expression,
with both downsampling and upsampling. All six candidate arrays matched the
legacy fixture. Candidate comparisons allow only 1e-12
absolute floating-point error and require identical dtypes.

`tests/image_runtime_smoke.py` imports the complete image-related FREA modules
with the real CARLA 0.9.16 client, then exercises imageio/native codec round trips,
matplotlib PNG output and actual FREA FFmpeg video export with frame decoding.
It does not connect to a simulator or load a checkpoint.
All seven full-module imports (including `CarlaRunner`), four image codec round trips, matplotlib PNG output
and three-frame FREA video export/decode passed. `pip check` passed for all 126
installed packages in the initial image-only validation, including editable FREA.
After adding the entrypoint dependency, all 142 installed packages passed `pip check`;
the 14 pixel snapshots, six resize arrays, full runtime smoke and four new actor
Traffic Manager port regression cases all passed again. The tested Linux CPU environment is
recorded in `tests/fixtures/pillow-validation-environment.txt`.

The official Pillow 13.0.0 test tree at
`293218ad059d994baf1777d882911bed6a49704a` was tested against its released wheel:
JPEG, libtiff, TIFF, PDF, PDF parser, GIF and morphology modules. Result:
**492 passed, 6 skipped**, with a 2 GiB address-space limit and 20-second per-test
timeouts. Skips concern optional djpeg, WEBP-in-TIFF support, Windows and extra
image files. These include security regressions for PDF trees/trailers,
TIFF ColorMap and zero-width morphology. This is not an ASan build or a test of
every possible codec/library combination.

Run project tests from the repository after installing its dependencies and client:

```bash
mkdir -p runs/pillow-check/tmp runs/pillow-check/cache
export TMPDIR="$PWD/runs/pillow-check/tmp" TMP="$PWD/runs/pillow-check/tmp" TEMP="$PWD/runs/pillow-check/tmp"
export XDG_CACHE_HOME="$PWD/runs/pillow-check/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export PYTHONPATH="$PWD:$CARLA_ROOT/PythonAPI/carla${PYTHONPATH:+:$PYTHONPATH}"
python tests/pillow_regression.py --compare tests/fixtures/pillow-10.3.0.json
python tests/resize_regression.py --compare tests/fixtures/resize-0.19.3.npz
python tests/image_runtime_smoke.py
```

For historical experiment reproduction use the original code revision, environment
and CARLA release in isolation. Before interpreting new training/evaluation scores,
run fixed-seed scenes and checkpoint inference in the new environment; CPU image
regressions do not establish identical vehicle physics, semantic labels, planner
behavior or GPU numerics across CARLA/Python versions. No training, checkpoint inference or comparative benchmark was performed. The
bounded functional smoke below verifies execution, not old/new score equivalence.


## Real CARLA closed-loop smoke (2026-10-26)

`tests/carla_closed_loop_smoke.py` starts its own existing CARLA 0.9.16 binary,
reserves common per-GPU and per-port flock leases, refuses busy GPUs/ports, checks
the actual Vulkan GPU UUID, and holds leases until its simulator and client exit.
The client is CPU-only and calls the real `CarlaRunner`, `VectorWrapper`,
`CarlaEnv`, behavior policy and scenario policy in evaluation mode. No learned
policy, checkpoint or training call is used. It uses Town05, Scenario9 route 8,
seed 2026, 0.1-second synchronous ticks and 128x128 RGB/semantic cameras.
FREA creates its usual minimum of five background vehicles even at traffic
intensity zero; these and the ego remain confined to the owned simulator.

Successful configuration: **default Epic quality**, GPU 4, RPC/stream/secondary/TM
ports 24000/24001/24002/24006, four CPU cores for the simulator, a two-thread CPU
client, and a 60-second client RPC timeout for cold starts.

- 60 observation/control/tick steps, 6.0 simulated seconds, frame delta 60.
- Ego path length 30.9009 m, maximum speed 8.4320 m/s, no ego collision.
- 55 distinct RGB image hashes; real semantic frames and Pillow mask compositing.
- Camera/mask PNG decoded pixels matched the saved arrays; video decoded 60 frames.
- Scene vehicle/sensor cleanup left zero actors. Both owned child processes exited
  before GPU/port leases were released; the simulator shutdown needed the bounded
  SIGKILL fallback after SIGTERM. Other tasks' processes were not signalled.
- All four actor control-transfer port regressions passed, covering background
  activation, controlled-vehicle selection, truncation and goal completion.

The earlier **Low-quality** profile crashed at the first camera warm-up tick on
this installation. GDB on a second idle GPU reproduced a native null
`VertexDeclaration` in `FVulkanVertexInputStateInfo::Generate`, called by Vulkan
PSO creation. This happened before Pillow image handling. The same full functional
smoke passed using Epic quality. This is a per-instance launch-profile limitation,
not evidence requiring a host driver change. Do not use this test result to claim
Low quality is supported or that historical sensor semantics/scores are identical.
No shared CARLA files, drivers, CUDA installations or other environments were
modified to obtain the pass.

The successful command, after setting project-owned TMP/XDG/cache paths and
CARLA's PythonAPI/carla in PYTHONPATH, was:

```bash
python tests/carla_closed_loop_smoke.py \
  --carla-root /home/keyuchen/data/carla_0.9.16 \
  --gpu 4 --port 24000 --quality Epic --steps 60 \
  --output-dir "$PWD/runs/carla-smoke-20261026/attempt-08-epic"
python -m unittest discover -s tests -p test_traffic_manager_ports.py
```

Use a **new** output directory on every invocation. The launcher never attaches to
an existing simulator and requires the shared coordination lock files used on
this server. It is intentionally opt-in and is not part of automatic unit-test
discovery. Failed-run diagnostics and the successful JSON/configuration, two PNGs,
short video, environment list and logs are retained locally under the run directory.
Temporary debugger/runtime homes, shader/build caches and intermediate files are
removed after all workload processes exit.
