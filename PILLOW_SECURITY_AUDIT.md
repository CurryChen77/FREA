# Pillow security remediation: blocked by the Python 3.8 environment

Audit date: 2026-10-26. Source revision: `455c5d078c240d81fb11902c0c47c3e7aef9facc`.

**Status: vulnerabilities remain unresolved. This change adds an audit and regression
coverage, not a dependency fix. Do not dismiss Dependabot alerts on this basis.**

## Official findings

| CVE | Upstream affected range | First upstream fix | Trigger / impact |
| --- | --- | --- | --- |
| CVE-2026-112403 | <=12.3.0 | 13.0.0 | JPEG draft/thumbnail downscaling with lossless SOF3 and libjpeg-turbo >=3; native heap overflow |
| CVE-2026-112897 | <=12.3.0 | 13.0.0 | Short JPEG-compressed TIFF strips/tiles decoded through libtiff can expose uninitialized heap bytes as pixels |
| CVE-2026-112895 | 5.1.0–12.3.0 | 13.0.0 | Parsing an existing PDF during append can exhaust stack/CPU/memory through page trees or trailer chains |

Pillow 10.3.0 falls in all three published version ranges. The JPEG issue's actual
reachability also depends on the linked JPEG library and use of scaling before load.
Its individual advisory is rated **High**; the 13.0.0 release notes group JPEG
decoder overflows under Critical. The reported alert rating should therefore be
kept distinct from the individual upstream advisory's rating.

Primary sources:

- https://github.com/python-pillow/Pillow/security/advisories/GHSA-8g5g-8vmr-9782
- https://github.com/python-pillow/Pillow/security/advisories/GHSA-q45j-58j2-45rg
- https://github.com/python-pillow/Pillow/security/advisories/GHSA-p4wg-p73j-c528
- https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html
- https://pillow.readthedocs.io/en/stable/installation/python-support.html
- https://github.com/python-pillow/Pillow/blob/13.0.0/pyproject.toml

The connected GitHub tools do not expose private Dependabot alert endpoints; the
provided alert URL was rejected as an unsupported endpoint. The three CVEs above
were verified against public upstream advisories, not against the repository's
complete alert inventory. Other Pillow alerts must be reconciled before claiming
complete remediation. No existing Pillow PR was returned by repository PR search.

## Why a one-line dependency upgrade cannot safely be applied

`README.md` creates Python 3.8 and `setup.py` enforces `>=3.8,<3.9`.
Pillow 13.0.0 declares `requires-python = ">=3.11"`. Pillow 10.4 is the last
series supporting Python 3.8 and does not fix these advisories. Pillow 11/12 also
do not fix all three. Upstream says fixes should not be expected to be backported.

The minimum version-only candidate would be:

```diff
-Pillow==10.3.0
+Pillow==13.0.0
```

**This diff is intentionally not applied.** There is no compatible intersection
between the project's declared Python range and the fixed upstream Pillow release.
An environment-marker fallback to 10.x would leave the actual experiment runtime
vulnerable. Ignoring Requires-Python is not a supported remediation.

A Python migration also requires reviewing CARLA 0.9.13's Python client and the
existing NumPy 1.24.4, SciPy 1.10.1, scikit-image 0.19.3, moviepy 1.0.3,
matplotlib 3.6.3 and torch 2.4.1 pins. Native wheels and removed Pillow APIs in
transitive image paths need validation. The project installation itself rejects
Python 3.11. No broad dependency upgrade or packaging-range change is included.

## Code reachability

The four Python files importing PIL use:

- `frea/agent/agent_utils/viz_tokens_bev.py`: in-memory L/RGB images, polygon/line
  drawing, NumPy conversion and PNG output; imported by the PlanT dataset.
- `frea/agent/agent_utils/explainability_utils.py`: BGRA/RGB conversion, RGBA
  masks and alpha paste; called by the CARLA environment.
- `frea/agent/PlanT/PlanT_agent.py`: BEV drawing, ImageOps borders and PNG output.
- `frea/agent/expert/nav_planner.py`: in-memory route debug drawing.

No direct production `Image.open`, `draft`, `thumbnail`, TIFF decode or PDF append
call was found in tracked Python source. This reduces demonstrated direct exposure;
it does not make the installed dependency safe. Plotting/imageio/scikit-image and
user scripts can introduce indirect file decoding. `run_util.py` imports moviepy's
FFmpeg writer, not its image-resize helper. Those complete transitive paths were
not executed. Treat external images/documents as untrusted even if their extension
looks like a supported format; extension checks alone are insufficient.

## Regression evidence and reproduction

`tests/pillow_regression.py` extracts and executes the actual image helper bodies
with AST, avoiding unrelated CARLA/torch imports. No helper logic is reimplemented.
It compares 14 deterministic output-array fingerprints: BEV extent/speed modes,
semantic-mask foreground/background compositing, route dots, and PlanT red/green
borders with PNG save/reload. Compare decoded pixels rather than compressed PNG
bytes, which may change with encoder versions.

Baseline: Python 3.11.17, Pillow 10.3.0, NumPy 1.24.4, OpenCV 4.10.0.84.
Candidate: same environment, only Pillow changed to 13.0.0.
Result: **all 14 snapshots passed and matched byte-for-byte**.
The same fingerprints also passed in Python 3.8.20 with Pillow 10.3.0 and the
same NumPy/OpenCV pins, establishing the legacy image-helper baseline.

```bash
# Use disposable environments, never an existing experiment environment.
python3.11 -m venv /tmp/frea-pillow-check
/tmp/frea-pillow-check/bin/pip install Pillow==10.3.0 numpy==1.24.4 opencv-python==4.10.0.84
/tmp/frea-pillow-check/bin/python tests/pillow_regression.py --compare tests/fixtures/pillow-10.3.0.json
/tmp/frea-pillow-check/bin/pip install --no-deps Pillow==13.0.0
/tmp/frea-pillow-check/bin/python tests/pillow_regression.py --compare tests/fixtures/pillow-10.3.0.json
```

A pip binary-download compatibility check for Python 3.8 rejected Pillow 13.0.0,
consistent with its official Python metadata. Passing image helper tests does not
establish full application compatibility or security coverage. No CARLA closed-loop
run, checkpoint inference, full dependency resolution, GPU test, or native security
regression suite was run.

## Required path to a complete fix

1. Preserve the original environment and record its packages/native codec versions
   and baseline experiment outputs before any migration.
2. In a separate environment, validate Python 3.11 and a matching CARLA client,
   then resolve the existing pins with the fewest necessary changes. Pin Pillow
   exactly to 13.0.0 after that compatibility work, rather than using an open range.
3. Run these pixel regressions, full-module imports, plotting/video export, and a
   fixed-seed CARLA/checkpoint smoke evaluation with the same input scenes. Check
   numerical outputs as well as execution success.
4. Reconcile the complete Dependabot inventory and rescan the resulting environment.

If Python 3.8 must be retained, a separately maintained and tested native backport
is needed. This audit does not supply or validate such a fork. Until a complete fix
is available, keep experiments isolated and restrict processing to trusted local
data; these are temporary exposure controls, not vulnerability fixes.
