Performance benchmarking
========================

The suite in ``tests/benchmarks`` measures Z-Rad computation, independently of
correctness-test duration. Use operation benchmarks to locate a change and the
IBSI workflows to assess its effect on realistic processing. Production code
and existing correctness tests are unchanged.

Install and run
---------------

Install the normal test extra (also included in ``dev``)::

    python -m pip install -e '.[test]'

``pytest-benchmark>=5.3.0`` is intentional: 5.3.0 was the current stable release
checked on 2026-09-11 and includes pytest 9 compatibility and current xdist
detection fixes. Candidate percentage changes are calculated explicitly by this
framework, independently of pytest-benchmark's file ordering. There is no
exact pin. ``threadpoolctl>=3.5`` controls supported BLAS/OpenMP backends, including
SciPy's bundled OpenBLAS. Neither dependency is required by ordinary Z-Rad users.
See the `pytest-benchmark changelog
<https://pytest-benchmark.readthedocs.io/en/latest/changelog.html>`_.

Normal tests retain their coverage and xdist defaults::

    python -m pytest
    python -m pytest -m unit
    python -m pytest -m integration

``pytest.ini`` sets ``--benchmark-skip``. The plugin skips benchmark tests before
fixtures run, so ordinary tests neither generate benchmark images nor extract
benchmark datasets. Benchmarks do not have ``unit`` or ``integration`` markers.

For timing, the recommended portable launcher sets native thread environment
variables **before** importing numerical libraries, then invokes native pytest::

    # Fast suite, no saved results
    python tests/benchmarks/run.py -m 'not benchmark_slow'

    # Complete suite, including slow scaling, I/O, pipeline, and IBSI
    python tests/benchmarks/run.py

    # IBSI only; or use -m benchmark_slow for every expensive workload
    python tests/benchmarks/run.py -k ibsi

The equivalent direct pytest commands are::

    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov -m 'not benchmark_slow'
    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov
    python -m pytest tests/benchmarks/test_benchmark_ibsi.py --benchmark-only -n 0 --no-cov

``--benchmark-only`` overrides the normal native skip. ``-n 0`` overrides
``-n auto``; competing pytest workers would otherwise measure CPU contention.
``--no-cov`` overrides the coverage configuration; instrumentation changes the
work being measured. The suite rejects intentional timing with workers, coverage,
explicit cProfile/pytest-memray profiling, disabled garbage collection, or active
Python tracing/profiling. Do not wrap these commands in external profilers.

Groups and exact measured regions
---------------------------------

Each factory in ``workloads.py`` prepares inputs and returns a callable plus a
validator and, where needed, per-round setup. ``measure`` invokes only the
operation callable inside ``benchmark.pedantic``'s timed region. Per-round setup
runs before each warmup and measured call, outside timing. Validation happens
after measurement. The shared definitions also power memory measurement without
importing the timing fixture.

.. list-table:: Workloads
   :header-rows: 1
   :widths: 18 44 38

   * - Group
     - Measured operation
     - Outside timing
   * - image
     - ``Image.resample_to_target`` with linear interpolation at three sizes;
       includes the API's array/SimpleITK conversions and output construction
     - Source and target grid construction
   * - preprocessing
     - ``ImageResampler.apply`` (linear and B-spline), linear mask resampling;
       separately intensity ROI construction, range/outlier re-segmentation,
       and 32-bin texture discretization
     - Synthetic image/mask generation; resampler construction; preceding ROI
       preparation for the individual re-segmentation/discretization cases
   * - filtering
     - ``apply`` for 3D mean (support 5), LoG (sigma 2 mm, cutoff 4), db3 HHL
       wavelet (level 1), and first-order Riesz-LoG; reflect padding
     - Image generation and filter construction. Metadata preparation and array
       conversions performed by ``apply`` remain included
   * - radiomics
     - Complete fresh-image ``families='all'`` extraction; six texture families together;
       and selected Moran's I/Geary's C, using 3D/MERG aggregation
     - Intensity-mask building, re-segmentation, texture and IVH discretization,
       extractor construction; local-means result-cache reset before each complete
       extraction round. Extractor-internal mask validation/copying remains included
   * - pet_suv
     - Enhanced PET ``_enhanced_suv_array`` on IBSI-SUV DRO_7_0_0, including
       metadata interpretation and SUVbw calculation
     - ZIP extraction, DICOM reading, and initial pixel decoding. The decoded
       pydicom pixel cache is deliberately warm
   * - pet_suv (I/O)
     - ``Image.from_dicom`` loading and converting conventional DRO_0_0
     - ZIP extraction and reference-mask loading. Filesystem/OS caches are warm;
       this is explicitly marked ``benchmark_io`` and ``benchmark_slow``
   * - pipeline
     - Image/mask resampling to 1.5 mm, LoG, ROI building, re-segmentation,
       texture/IVH discretization, complete radiomics extraction
     - Synthetic inputs, pipeline and extractor construction
   * - ibsi
     - IBSI I configuration C: linear image/mask resampling to 2 mm, CT rounding,
       range [-1000, 400], texture width 25, IVH width 2.5, 3D/MERG extraction
     - CT/RTSTRUCT loading, extraction, reference CSV loading, published-tolerance
       checks. Checks use ``ibsi_helpers`` and the same selection as correctness tests
   * - ibsi
     - IBSI II phase II configuration 3.B: 1 mm B-spline image/linear mask,
       CT rounding, 3D LoG sigma 1.5 mm, ROI/range/width-25 preparation, extraction
     - Same setup/validation exclusions as IBSI I. The existing correctness
       workflow uses **2D/AVER features after 3D filtering**; this is preserved

IBSI uses the repository's CT phantom and GTV-1 RTSTRUCT, reusing the session
extraction fixture. Configuration C covers calibrated-intensity full extraction;
3.B covers resampling, filtering and filtered-image feature extraction. These two
signals avoid duplicating the complete IBSI correctness matrix. They are always
slow benchmarks, irrespective of speed on a particular machine.

The Enhanced PET helper is a deliberate private-API benchmark because conventional
SUV conversion nests its calculations inside a function that rereads files.
No production API was changed to expose it. Historical versions without this
helper cannot use this harness unchanged.

Sizes and scaling
-----------------

Array shapes are **(z, y, x)**; geometry spacing is **(x, y, z)**.

.. list-table:: Deterministic synthetic inputs
   :header-rows: 1

   * - Size
     - Resampling/filtering volume
     - Radiomics volume
   * - small
     - 32 x 96 x 96 (294,912 voxels)
     - 16 x 24 x 24 (9,216 voxels)
   * - medium
     - 64 x 128 x 128 (1,048,576 voxels)
     - 24 x 40 x 40 (38,400 voxels)
   * - large (slow)
     - 96 x 192 x 192 (3,538,944 voxels)
     - 40 x 64 x 64 (163,840 voxels)

All use float64 data, spacing (1, 1, 2) mm, a fixed RNG seed (20260911), smooth
structure plus nonconstant noise, and an ellipsoidal ROI covering approximately
23% of the volume. Actual voxel count/fraction is stored. The image sizes exercise
megabyte-scale allocation and 3D computation; ROI-specific volumes bound expensive
texture work while scaling from about 2,000 to 38,000 ROI voxels. These are selective
patch/ROI workloads, not a claim to represent every scanner's full field of view.
Real CT workflows supply the complementary larger, irregular workload.

Stable ``extra_info.workload_id`` identifiers match timing and memory results.
Metadata records shapes, ROI size/fraction, spacing, dtype, parameters, seed,
aggregation, requested/observed thread controls, dependencies, loaded Z-Rad path,
and a hash of the benchmark Python files. Change the suite version and workload
identifier if the measured region, data distribution, or parameters change.
Do not compare incompatible workloads just because test names match.

Rounds, state, threads, and statistics
--------------------------------------

Most operations run seven measured calls; expensive scaling/radiomics/PET cases
use five; pipeline and IBSI use three. Each has one unmeasured warmup and one
iteration per round. This bounds the costly 3D work. Initial measurements found
the fast suite took seconds, whereas a single IBSI II call took tens of seconds;
automatic calibration of that workflow would provide little benefit. Three
IBSI rounds give only a preliminary distribution, not a precise confidence bound.
To characterize noise, repeat entire runs and, if justified, increase the specific
workload's rounds. Native ``--benchmark-min-rounds`` does not override pedantic rounds.

Input arrays are read-only; preprocessing/filtering return new outputs, and the
extractor builds a fresh context and feature groups per call. Complete radiomics
extraction additionally resets ``_LOCAL_MEANS_CACHE`` in per-round setup: this
module-level cache otherwise reuses local means by image-array identity despite
fresh extraction contexts. Both image and support convolutions execute in every
measured round. Cache clearing, ROI preparation and extractor construction stay
outside timing. Inspection found no other module-level or identity-based
radiomics result cache affecting this workload.

Selected filters do not cache output volumes. LoG updates spacing-derived scalar
state in ``apply``; the same geometry is used every round. Decoded PET pixels are
cached explicitly in setup. Warmup absorbs library initialization; complete
radiomics keeps image-derived results uncached. These runs do not measure cold
interpreter/import or cold filesystem latency. All allocations and Python object
construction performed by the operation stay included. Garbage collection stays enabled.

Suite version 2 renames complete extraction to ``test_complete_fresh_extraction``
and uses workload IDs ``radiomics/all_fresh/{size}``. Earlier
``test_complete_extraction`` / ``radiomics/all/{size}`` timing results were warmed
by the local-means result cache and are **invalid as fresh-extraction references**.
Do not merge or compare those timing series. Texture/spatial workload identities
are unchanged. Earlier RSS/allocation measurements already used fresh processes
without warmup; their methodology is unaffected, although the shared complete
extraction workload ID is now renamed in memory output too.

SimpleITK and OpenCV have explicit thread APIs. OpenCV GCD builds require
``setNumThreads(0)`` to disable parallel regions; other backends use 1. The
reported serial state is checked, and prior settings are restored afterward. NumPy/SciPy BLAS and OpenMP pools
are limited with ``threadpoolctl`` after imports; SciPy FFT uses ``set_workers(1)``.
The launchers additionally set ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
``MKL_NUM_THREADS`` and ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`` to 1 before import.
These cover the numerical backends used by the installed dependencies. PyWavelets
and the selected ndimage kernels have no separate thread-pool setting here.
Joblib batch parallelism is not invoked. Thread limits are restored after pytest.

On macOS the launchers also set ``VECLIB_MAXIMUM_THREADS=1``. Apple Accelerate is
not introspected by threadpoolctl, so its limit is recorded as **unverified**;
an empty native-pool list does not prove serial BLAS execution. Direct pytest
users on such systems should export this variable before starting Python. Prefer
the canonical Linux/OpenBLAS environment for controlled comparisons. See
`threadpoolctl's Accelerate limitation <https://github.com/joblib/threadpoolctl/issues/135>`_.

Use **median** as the headline, retaining mean, min/max, standard deviation, IQR,
rounds and iterations in native JSON. ``--benchmark-json`` includes round samples;
use ``--benchmark-save-data`` with saved runs to retain them there too. A minimum
is not a regression verdict. IQR is within-run spread, not a confidence interval
for changes between runs. Rerun noisy changes on an idle machine; compare repeated
runs in both execution orders. Do not infer improvements from differences similar
to normal variability. No timing or memory failure thresholds are installed.

Saving and comparing references
-------------------------------

Native JSON is the sole timing format. No timing database or custom baseline
updater is introduced. For example::

    python tests/benchmarks/run.py -m 'not benchmark_slow' --benchmark-save=master-accepted --benchmark-save-data
    python tests/benchmarks/run.py -m 'not benchmark_slow' --benchmark-autosave --benchmark-save-data
    pytest-benchmark list
    # Replace 0001 with the explicit accepted reference ID from the listing
    python tests/benchmarks/run.py -m 'not benchmark_slow' --benchmark-compare=0001 --benchmark-save=pr-candidate

Or use explicit files (create parent directories first)::

    mkdir -p reports/benchmarks
    python tests/benchmarks/run.py -m 'not benchmark_slow' --benchmark-json=reports/benchmarks/current.json
    python tests/benchmarks/compare_results.py --current reports/benchmarks/current.json --master reports/benchmarks/master.json --release reports/benchmarks/release.json
    pytest-benchmark compare reports/benchmarks/master.json reports/benchmarks/current.json --columns=median,iqr,mean,stddev,min,max,rounds,iterations

``compare_results.py`` uses explicit reference roles and reports
``100 * (current_median - reference_median) / reference_median`` separately for
master and release. Thus 1 s to 2 s is **+100% regression**, 2 s to 1 s is
**-50% improvement**, and equal values are **0% unchanged**. These labels describe
direction, not statistical significance. Reference/current IQR values are shown
as spread, without interpreting IQR changes as performance improvements.
Workloads are matched by ``extra_info.workload_id`` (or full test name for JSON
without that field); unmatched workloads are reported as unavailable. A zero or
invalid reference timing produces no percentage verdict.

Do not use ``pytest-benchmark compare --between`` for reports of candidate changes:
pytest-benchmark 5.3.0 sorts filenames and may select ``current.json`` as its
baseline, regardless of argument order. Previously generated summaries using that
command must be regenerated from the raw JSON. The raw timing samples are not
changed by the reporting fix; the complete-extraction exception is described above.
The ordinary native comparison command preserves all the statistics. With no explicit reference,
``--benchmark-compare`` uses the latest saved run, which may be a candidate:
**always name the accepted reference when making regression decisions**.

* **Master reference:** accepted master SHA, updated deliberately after accepted
  changes reach master. A slower passing PR cannot redefine it.
* **Release reference:** a specific published tag/SHA, fixed until the next
  release reference is intentionally created. Retain its environment and harness.
* **Current/PR:** a candidate, saved separately. Never overwrite either reference.

Names like ``master-accepted`` are labels, not an automatic branch checkout.
A named save must actually execute the intended revision. Do not run another
checkout's tests while an editable installation still points at the current code.
Long-lived reference JSON belongs in a controlled artifact archive; the repository
ignores generated JSON, ``.benchmarks``, reports and profiling captures.

Same-machine comparisons and CI
-------------------------------

A portable comparison runner resolves refs once, archives each tracked revision
into a fresh temporary source tree, installs it **non-editably** in a separate
virtual environment, and runs a single copy of the **current benchmark harness**
outside those source trees. It asserts that ``zrad.__file__`` belongs to the
intended environment. This avoids stale editable installs, bytecode/build products,
and accidental benchmark-definition differences. Each revision is run sequentially
on the same machine and Python interpreter with identical thread controls.

For local committed changes::

    git fetch origin master --tags
    python tests/benchmarks/compare_revisions.py --master origin/master --release v26.8.0 --current HEAD --output reports/benchmarks/comparison-001

Replace the example release tag with the latest **published release** you intend
to evaluate; the newest version-sorted tag is not necessarily a release. Use
``--full`` for slow/IBSI comparisons. Uncommitted production edits are excluded by
``git archive``; use the ordinary launcher to time working-tree edits. The current
working-tree harness is used intentionally. The output directory must be new,
protecting earlier results. Source trees and environments are removed afterward.

Dependencies are resolved separately from each revision's declared requirements.
All use the same benchmark tools, but compatible version ranges can resolve to
different libraries. ``*-dependencies.txt`` and per-run metadata expose this
confounder. Use ``--constraints /absolute/path/constraints.txt`` to supply a common
set when comparing algorithm changes with fixed dependencies. No silently forced
incompatible dependency set or persistent shared editable environment is used.

The separate ``benchmark.yml`` workflow uses **ubuntu-24.04, Python 3.12** on one
GitHub-hosted runner. It runs on PRs/pushes to master and manual dispatch. On PRs,
``HEAD`` is GitHub's tested merge candidate; ``origin/master`` is resolved at job
checkout time. On master pushes, current and master may be identical (a useful
noise observation, not a claimed speedup). The GitHub release API resolves the
latest published stable release, then the runner freezes its SHA for that job.
It does not alter any saved accepted/release reference. Manual dispatch can opt
into the full suite. The compatibility/coverage workflow remains separate.

Artifacts retained for 30 days contain valid ``current.json``, ``master.json``,
and ``release.json`` where available, installation/test logs, dependency lists,
a status/commit manifest and explicitly reference-relative median/IQR comparison output. The comparison
also appears in the job summary, with current/master and current/release
percentage changes shown separately where each reference is valid. GitHub-hosted hardware and background load vary;
the runner image label does not pin hardware or every OS package. Results are
informative, without small-percentage gates. Measurement/test failures in the
candidate still fail the job. No privileged ``pull_request_target`` or PR-comment
publishing is used.

Old revisions may lack the current API or fail current validation. They are
reported as incompatible/failed, with logs; partial failing results are labelled
``INVALID`` and excluded from comparison. Candidate failures are never silently
skipped. In particular, v26.8.0 lacks RieszLoG and the Enhanced PET helper required
by this harness; its full-suite comparison is unavailable. The initial framework
prioritizes PR/master comparisons instead of inventing historical adapters or
claiming unequal workflows are equivalent. Release tracking starts when a release
supports this harness (or a reviewed common-workload adapter is added). Never
interpret an absent row/reference as unchanged performance.

Memory: separate metrics and processes
--------------------------------------

Peak RSS is measured without pytest or timing instrumentation::

    python tests/benchmarks/memory.py --mode rss --repeats 3 --output reports/benchmarks/rss-001.json
    python tests/benchmarks/memory.py --mode rss --workload ibsi_i --workload ibsi_ii --output reports/benchmarks/rss-ibsi.json

The six workloads are large B-spline resampling, medium Riesz-LoG, large complete
radiomics, large spatial statistics, IBSI I C and IBSI II 3.B. Each repetition
executes in a **fresh subprocess**, with no warmup. Archive extraction occurs in
the parent before any measured child starts. ``resource.getrusage(RUSAGE_SELF)``
is sampled after computation and before validation/report serialization. Linux
KiB values and macOS byte values are normalized to bytes.

``peak_rss_bytes`` is the **process-lifetime maximum resident set through completion
of the workload**, including interpreter/native imports, loaded inputs and setup.
``setup_peak_rss_bytes`` is the high-water mark immediately before computation;
it is neither current RSS nor a number to subtract to obtain temporary allocation
size. Setup may itself dominate a small operation. The maximum is not polluted by
previous workloads because every sample has its own process/PID. Children do not
spawn batch workers. A process tree would require a different measurement design.
This backend supports Linux/macOS; Windows timing remains usable and optional
profiling dependencies are excluded there.

For allocation attribution, install the optional profiling extra::

    python -m pip install -e '.[test,profiling]'
    python tests/benchmarks/memory.py --mode memray --workload ibsi_ii --output reports/benchmarks/alloc-ibsi-001.json
    python tests/benchmarks/memory.py --mode memray --native --workload spatial --output reports/benchmarks/alloc-spatial-001.json
    python -m memray stats reports/benchmarks/alloc-spatial-001-spatial-0.memray
    python -m memray flamegraph reports/benchmarks/alloc-spatial-001-spatial-0.memray

Memray's tracker wraps **only one operation after input preparation**, in its own
fresh child. It captures Python and native heap allocations. ``--native`` adds
native stack attribution and extra overhead; it is optional. The reported
``allocation_high_water_bytes`` is Memray's peak simultaneously live **tracked
allocations during the operation**, not peak RSS/RAM. Preexisting input allocations
are not counted. Validation and setup are outside the tracker. No timing JSON is
created and RSS is not reported from an instrumented run. Do not combine those
runtimes with timing results. Native-symbol reports should be generated on the
capture machine; large captures are ignored by git. See
`Memray native tracking <https://bloomberg.github.io/memray/run.html>`_.

pytest-memray was investigated but is not installed: its per-test instrumentation
would include fixture/test work and complicate timing separation. Direct Memray
tracking provides the desired narrower region with one optional dependency.
Memray remains limited to Linux/macOS and is never a normal runtime dependency.

Memory JSON has ``schema_version: 1`` and a ``measurements`` list: stable workload
ID, commit, working-tree dirty flag, PID, timestamp, Python/platform, workload/environment metadata, and
one mode-specific metric. All repetitions remain separate; no timing schema is
repurposed. Existing output paths are refused. Peak RSS is suitable for an initial
platform-specific longitudinal series; characterize variance before gating it.
Memray results and captures are diagnostic allocation evidence, not substitute RSS.

Initial observations and extending the suite
--------------------------------------------

Before the suite-version-2 cache correction, local verification on macOS/Apple Silicon,
Python 3.14.6, measured 20 fast
cases in approximately 6.6 seconds and 11 slow cases in approximately 111 seconds
(warmups/setup included in suite duration). IBSI medians were approximately 0.81 s
and 24 s. Two fresh-process RSS observations ranged around 300 MiB for resampling,
264 MiB Riesz-LoG, 241 MiB radiomics, 223 MiB spatial, 361 MiB IBSI I and 1.28 GiB
IBSI II. These are development observations, **not versioned baselines**; Accelerate
threading is unverified and dependencies/hardware differ from CI. Recheck on your
machine. First-time ZIP extraction also changes total suite duration, not operation
timing.

When adding a benchmark:

1. Inspect the production path, mutation, retained buffers, caches and threading.
   Select an operation that contributes material workload cost.
2. Add a deterministic factory with a stable workload ID and explicit parameters.
   Construct/load inputs and operators before returning the measured callable.
3. Time only the intended region. Put output checks in the validator. Reuse
   published references where available, without timing reference loading.
4. Make repeated calls independent. Use read-only input arrays where possible;
   for mutating APIs use ``benchmark.pedantic(setup=...)`` to clone inputs outside
   each timed round. Never merely reset once before a multi-iteration round.
5. Select meaningful sizes; measure their time/memory before enabling them by
   default. Mark costly cases ``benchmark_slow`` and filesystem cases ``benchmark_io``.
6. Include the explicit ``benchmark`` fixture in each test so native exclusion
   works. Add a group, call ``measure``, run the test, and inspect the saved JSON.
7. Add the shared factory to the memory CLI only if its allocations matter.
   Keep memory instrumentation out of authoritative timing runs.

Laws/Gabor, Simoncelli, every filter/IBSI configuration, pure NIfTI I/O and joblib
batch scaling were deliberately omitted from the initial selective matrix.
Mean/LoG, separable wavelets and Riesz cover different computational paths;
Gabor's kernel cache would need an explicitly chosen warm/cold policy. Batch work
mixes filesystem and parallel scheduling costs requiring a separate methodology.
The synthetic pipeline is the initial system-level signal alongside real IBSI.

Useful future work includes a stable dedicated runner, repeated/counterbalanced
revision order, archived environment constraints, characterized per-workload noise
budgets, a reviewed common subset for older releases, and larger or sparse ROI
shapes to exercise spatial algorithm crossover. These are future improvements,
not claims made by the current implementation.
