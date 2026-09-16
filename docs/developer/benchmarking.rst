Performance benchmarking
========================

The benchmarks in ``tests/benchmarks`` measure Z-Rad execution time and memory
use. Use individual operation benchmarks to identify where performance changed,
and IBSI workflows to assess the effect on complete processing workflows.
Setup and validation run outside the timed region.

Quick start
-----------

Run these commands from the repository root. Install the test dependencies
(also included in the ``dev`` extra), then run the standard suite::

    python -m pip install -e '.[test]'
    python tests/benchmarks/run.py --suite standard

The launcher sets thread limits before importing numerical libraries and runs
pytest with parallel workers and coverage disabled. Ordinary test runs skip
benchmarks before their fixtures run; see :doc:`testing` for test commands.

Choose a suite
--------------

Timing and memory measurements use the same suite definitions and workload IDs.

.. list-table:: Benchmark suites
   :header-rows: 1
   :widths: 15 10 75

   * - Suite
     - Cases
     - Coverage
   * - ``standard`` (default)
     - 100
     - Synthetic operation benchmarks and all 20 IBSI I workflows
   * - ``exhaustive``
     - 151
     - All standard cases, plus 33 IBSI II phase-I filter response maps and
       18 IBSI II phase-II feature workflows
   * - ``ibsi``
     - 71
     - Only the published IBSI cases: ``ibsi1``, ``ibsi2_phase1``, and
       ``ibsi2_phase2``

For broader coverage or a focused IBSI run::

    python tests/benchmarks/run.py --suite exhaustive
    python tests/benchmarks/run.py --suite ibsi

Save and compare results
------------------------

Compare committed revisions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the revision runner to compare committed changes with master on the same
machine::

    git fetch origin master --tags
    python tests/benchmarks/compare_revisions.py --master origin/master --current HEAD --output reports/benchmarks/comparison-001

The output directory must be new. The runner resolves each ref once, installs
it non-editably in a separate temporary virtual environment, and runs revisions
sequentially with the same Python interpreter and thread controls. It uses the
current working-tree benchmark harness for all revisions and checks that
``zrad.__file__`` points to the intended installation. Temporary source trees
and environments are removed afterward.

Uncommitted production changes are excluded. Use ``run.py`` to measure
working-tree changes. Pass ``--suite exhaustive`` for full coverage;
``--full`` is a deprecated alias for that option.

To include a release comparison, add ``--release RELEASE_TAG``, replacing
``RELEASE_TAG`` with a published release compatible with the current harness.
For example, v26.8.0 is incompatible because it lacks ``RieszLoG`` and
``Simoncelli``. The runner imports the workload module before selecting a suite,
so choosing a smaller suite does not avoid that incompatibility. Failed or
incompatible references are reported with logs; partial results are labelled
``INVALID`` and excluded from comparisons. Candidate failures fail the run.

Dependencies are resolved from each revision's requirements and may differ.
Check ``*-dependencies.txt`` and the run metadata. To hold dependencies fixed,
pass ``--constraints /absolute/path/constraints.txt`` with versions compatible
with all revisions being compared.

Save individual runs
~~~~~~~~~~~~~~~~~~~~

Timing results use pytest-benchmark's JSON format. To save a working-tree run::

    mkdir -p reports/benchmarks
    python tests/benchmarks/run.py --suite standard --benchmark-json=reports/benchmarks/current.json

If you already have an accepted ``master.json`` reference, compare it explicitly::

    python tests/benchmarks/compare_results.py --current reports/benchmarks/current.json --master reports/benchmarks/master.json

Add ``--release reports/benchmarks/release.json`` when a release reference is
available. For a side-by-side view of the underlying statistics, use::

    pytest-benchmark compare reports/benchmarks/master.json reports/benchmarks/current.json --columns=median,iqr,mean,stddev,min,max,rounds,iterations

You can also use pytest-benchmark's named saves. Run the first command at the
accepted master revision, then switch to the candidate before saving it::

    python tests/benchmarks/run.py --suite standard --benchmark-save=master-accepted --benchmark-save-data
    pytest-benchmark list
    # Replace 0001 with the accepted reference ID from the listing
    python tests/benchmarks/run.py --suite standard --benchmark-compare=0001 --benchmark-save=pr-candidate --benchmark-save-data

``--benchmark-json`` includes individual round samples. For named saves,
``--benchmark-save-data`` retains them. Always specify the reference ID:
``--benchmark-compare`` without an ID uses the latest saved run, which may be
another candidate.

Maintain references
~~~~~~~~~~~~~~~~~~~

* **Master reference:** an accepted master SHA, updated deliberately after
  accepted changes reach master. A passing PR does not automatically replace it.
* **Release reference:** a published tag/SHA, retained until you deliberately
  select a new release reference.
* **Current/PR:** a candidate, saved separately from both references.

Save names are labels; they do not check out revisions. Ensure that each run
executes the intended code, especially when using editable installations across
checkouts. Retain the reference's workload and environment metadata in a durable
artifact archive. Generated JSON, ``.benchmarks``, reports, and profiling captures
are ignored by git.

.. note::

   Create references with the version-1 suite. Development runs used changing
   workload definitions, and early complete-extraction timings reused cached
   local means. Those timings are invalid as fresh-extraction references.

Interpret results
-----------------

Use the median runtime as the main comparison metric. ``compare_results.py``
calculates each change relative to the explicitly selected reference::

    100 * (current_median - reference_median) / reference_median

A change from 1 s to 2 s is a **+100% regression**; 2 s to 1 s is a
**-50% improvement**. These labels describe direction, not statistical
significance. The interquartile range (IQR) describes spread within a run; it is
not a confidence interval for the difference between runs.

Retain mean, minimum, maximum, standard deviation, IQR, rounds, and iterations.
Repeat noisy measurements on an idle machine, running revisions in both orders.
Differences comparable to normal run-to-run variability are inconclusive.
The suite has no timing or memory regression thresholds.

Results are matched by ``extra_info.workload_id``, falling back to the full test
name when that field is absent. Check that workload definitions and environments
are compatible before comparing. Missing workloads are reported as unavailable;
zero or invalid reference timings produce no percentage verdict. Missing results
do not indicate unchanged performance.

.. warning::

   Use ``compare_results.py`` for candidate percentage changes. In
   pytest-benchmark 5.3.0, ``compare --between`` sorts filenames and may choose
   ``current.json`` as the baseline regardless of argument order. Regenerate
   summaries made with that command from the raw JSON.

Measure memory
--------------

Peak process memory
~~~~~~~~~~~~~~~~~~~

Resident set size (RSS) measures memory resident in RAM. Run the RSS suite
separately from timing, without pytest or profiling instrumentation::

    python tests/benchmarks/memory.py --mode rss --suite standard --repeats 3 --output reports/benchmarks/rss-standard-001.json

Use ``--suite ibsi`` or ``--suite exhaustive`` for broader IBSI coverage, or
select individual workload IDs::

    python tests/benchmarks/memory.py --mode rss --workload ibsi/i/c/3d/merg --workload ibsi/ii/phase_ii/3.b --output reports/benchmarks/rss-selected-001.json

Each repetition runs in a fresh subprocess without warmup. Archive extraction
happens in the parent before measurement. The child prepares inputs, runs any
setup hook, records the setup peak, executes the operation, and records the final
peak before validation and report serialization. Linux and macOS values are
normalized to bytes. This backend supports Linux and macOS; timing is also
available on Windows. RSS runs print grouped workload tables and save a Markdown
summary next to the JSON (for example, ``rss-standard-001.md``). Existing JSON
and Markdown output paths are refused rather than overwritten. Memray runs keep
their separate capture/manifest output and do not produce RSS summaries.

The two RSS fields have distinct meanings:

* ``peak_rss_bytes`` is the process-lifetime maximum through completion of the
  operation, including imports, inputs, and setup.
* ``setup_peak_rss_bytes`` is the process-lifetime maximum immediately before
  the operation. It is not current RSS; subtracting it from the final peak does
  not measure temporary allocations.

Setup may dominate small workloads. Separate processes keep previous workloads
from affecting the peak; child processes do not launch batch workers. RSS uses a
fresh process while timing uses warmup rounds, so the metrics describe different
execution conditions.

Allocation profiling
~~~~~~~~~~~~~~~~~~~~

Use Memray to investigate allocations in selected cases after RSS screening.
Install the optional profiling extra on Linux or macOS::

    python -m pip install -e '.[test,profiling]'
    python tests/benchmarks/memory.py --mode memray --native --workload radiomics/spatial/medium --output reports/benchmarks/alloc-spatial-001.json
    python -m memray stats reports/benchmarks/alloc-spatial-001-radiomics-spatial-medium-0.memray
    python -m memray flamegraph reports/benchmarks/alloc-spatial-001-radiomics-spatial-medium-0.memray

Memray tracks Python and native heap allocations during one operation in a fresh
child process. Input preparation, setup, and validation are outside the tracker.
``--native`` adds native stack attribution and overhead; omit it when those
stacks are unnecessary. Generate native-symbol reports on the capture machine;
see `Memray native tracking <https://bloomberg.github.io/memray/run.html>`_.

``allocation_high_water_bytes`` is the peak simultaneously live tracked
allocation size during the operation. It excludes preexisting inputs and is
not peak RSS. Instrumented runs produce neither timing JSON nor RSS measurements;
use their captures to diagnose allocations rather than compare execution times.

Memory result files
~~~~~~~~~~~~~~~~~~~

Memory JSON uses ``schema_version: 1`` and a ``measurements`` list. Each entry
includes a workload ID, revision and dirty flag, PID, timestamp, Python/platform,
workload/environment metadata, ``memory_methodology_version`` (1 for setup before
operation), and mode-specific measurements. Repetitions remain separate, and
existing output paths are refused.

Join ``measurements[].workload_id`` with timing ``extra_info.workload_id`` to
inspect both metrics for a case. Measure run-to-run variability on your platform
before setting memory failure thresholds.

Measurement methodology
-----------------------

Measured operations
~~~~~~~~~~~~~~~~~~~

Factories in ``tests/benchmarks/workloads.py`` prepare inputs and return an
operation, a validator, and optional per-round setup. ``measure`` times only the
operation through ``benchmark.pedantic``. Setup runs before each warmup and
measured call; validation runs afterward. Allocations, array conversions, and
object construction performed by the operation are included.

Timing and memory use the same case declarations in ``tests/benchmarks/cases.py``.
Published IBSI declarations in ``tests/ibsi_cases.py`` are also shared with
correctness tests. Registry tests check the 20/33/18 published case inventory.

.. list-table:: Workload boundaries
   :header-rows: 1
   :widths: 18 42 40

   * - Group
     - Measured operation
     - Setup and validation outside timing
   * - image
     - ``Image.resample_to_target``, including array/SimpleITK conversions
     - Source and target grids; geometry and background-fill checks
   * - preprocessing
     - Image/mask resampling, or individual ROI preparation,
       re-segmentation, and discretization operations
     - Synthetic inputs, resampler construction, and preceding ROI preparation
   * - filtering
     - Filter ``apply``, including Gabor kernel generation
     - Synthetic inputs, filter construction, and Gabor cache reset
   * - radiomics
     - Complete extraction, individual feature families, texture aggregation,
       or spatial statistics; includes extractor-internal mask validation/copying
     - ROI preparation, discretization, extractor construction, and local-means
       cache reset for complete extraction
   * - ibsi1
     - Configured preprocessing and radiomics extraction
     - Phantom/CT/RTSTRUCT loading, pipeline construction, and reference checks
   * - ibsi2_phase1
     - Filter ``apply`` for published response maps
     - Phantom/reference loading, filter construction, and response-map checks
   * - ibsi2_phase2
     - Configured resampling, filtering, ROI preparation, and radiomics extraction
     - CT/RTSTRUCT loading, pipeline construction, and reference checks

The operation cases cover source-derived and independent resampling grids,
isotropic and in-plane resampling, intensity ROI construction, range/outlier
re-segmentation, and fixed-bin-number/fixed-bin-size discretization. Filtering
covers Mean, LoG, db3 wavelets, Laws, Gabor, Simoncelli, and Riesz-LoG variants.
Radiomics covers all 11 feature families, IBSI texture aggregation paths, and
Moran's I/Geary's C. See the case registry and workload factories for individual
parameters and interpolation choices.

IBSI workflows use repository digital phantoms and the CT phantom, with GTV-1
RTSTRUCT for CT workflows. Pure NIfTI I/O and joblib batch scaling are outside
this computation-focused suite.

Synthetic inputs and metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Array shapes are **(z, y, x)**; geometry spacing is **(x, y, z)**.

.. list-table:: Synthetic input sizes
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

Inputs use float64 data, spacing (1, 1, 2) mm, a fixed random seed (20260911),
smooth structure with noise, and an ellipsoidal ROI covering approximately 23%
of the volume. These synthetic patches keep texture computation manageable;
CT workflows provide larger, irregular inputs. Complete extraction and spatial
statistics run only at medium radiomics size. Small and large radiomics inputs
are available to factories for future scaling studies.

Metadata records shapes, ROI size/fraction, spacing, dtype, parameters, seed,
aggregation, requested and observed thread controls, dependencies, loaded Z-Rad
path, and a hash of benchmark Python files. The current suite version is 1.
Change the suite version and workload ID when the measured region, input
distribution, or parameters change.

Rounds and cache policy
~~~~~~~~~~~~~~~~~~~~~~~

Most operations use seven measured rounds; expensive scaling/radiomics cases
use five, and published IBSI cases use three. Each has one unmeasured warmup and
one call per round. ``--benchmark-min-rounds`` does not override these explicit
pedantic round counts. Three rounds provide only a preliminary view of variability;
repeat entire runs and increase a workload's rounds when needed.

Inputs are read-only, preprocessing/filtering return new outputs, and extraction
creates a fresh context and feature groups per call. Complete radiomics extraction
also clears ``_LOCAL_MEANS_CACHE`` before every round so both image and support
convolutions execute each time. Gabor kernel caches are cleared before every
round so kernel generation remains measured. Memory runs use the same setup hooks
before their single operation.

LoG reuses spacing-derived scalar state with identical geometry each round.
The level-2 2D separable wavelet evaluates four rotations regardless of its
``rotation_invariance`` parameter; metadata records the effective count.
Warmup absorbs library initialization. Timing excludes cold interpreter/import
and filesystem startup latency. Garbage collection remains enabled.

Thread controls and direct pytest use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The launchers set ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
``MKL_NUM_THREADS``, and ``ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`` to 1 before
imports. After import, ``threadpoolctl`` limits supported BLAS/OpenMP pools,
including SciPy's bundled OpenBLAS; SciPy FFT uses ``set_workers(1)``.
SimpleITK and OpenCV use explicit thread APIs. OpenCV GCD builds use
``setNumThreads(0)`` to disable parallel regions; other backends use 1.
Reported serial state is checked, and prior settings are restored after pytest.
PyWavelets and the selected ndimage kernels have no separate thread-pool setting.

On macOS, launchers also set ``VECLIB_MAXIMUM_THREADS=1``. Apple Accelerate's
limit is recorded as **unverified** because threadpoolctl cannot inspect it.
An empty native-pool list does not establish serial BLAS execution. Prefer
Linux/OpenBLAS for controlled comparisons; see
`threadpoolctl's Accelerate limitation <https://github.com/joblib/threadpoolctl/issues/135>`_.

The following direct pytest commands select the same suites, but require you to
set the thread environment variables above before starting Python::

    # standard
    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov -m 'not benchmark_exhaustive'
    # exhaustive
    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov
    # ibsi
    python -m pytest tests/benchmarks --benchmark-only -n 0 --no-cov -m 'benchmark_ibsi'

``--benchmark-only`` overrides the normal ``--benchmark-skip`` setting;
``-n 0`` disables pytest workers and ``--no-cov`` disables coverage.
The suite rejects timing with workers, coverage, explicit cProfile/pytest-memray
profiling, disabled garbage collection, or active Python tracing/profiling.
Run timing commands without external profilers.

CI behavior
-----------

The separate ``benchmark.yml`` workflow uses Ubuntu 24.04 and Python 3.12 on one
GitHub-hosted runner. PRs and pushes to master run ``standard``; manual dispatch
selects any suite, and a weekly schedule runs ``exhaustive``. Exhaustive manual
and scheduled runs also produce an isolated-process RSS artifact for every case,
with one sample per case. The memory job summary shows all workloads in separate
grouped tables with peak and setup-peak RSS in MiB, using the automatically
generated Markdown file. Repeated local reports also show median peaks and the
observed range. To render an older JSON result manually, use::

    python tests/benchmarks/summarize_memory.py reports/benchmarks/rss-standard-001.json

On PRs, current is GitHub's tested merge candidate and master is resolved at job
checkout. On master pushes, the two may be identical, providing an observation
of measurement noise. The GitHub release API selects the latest published stable
release, whose SHA is fixed for the job. These comparisons do not update saved
accepted references. Incompatible releases are handled as described above.

Artifacts are retained for 30 days and include valid timing JSON where available,
installation/test logs, dependency lists, a status/commit manifest, and comparison
output. The job summary shows median changes relative to master and release,
with IQR values. GitHub-hosted hardware and background load vary, so use these
results as performance indicators. Candidate measurement or validation failures
still fail the job. Correctness and coverage checks run in a separate workflow.

Add a benchmark
---------------

1. Inspect the production path, mutation, retained buffers, caches, and threading.
   Select an operation that contributes material workload cost.
2. Add a deterministic factory with a stable workload ID and explicit parameters.
   Construct/load inputs and operators before returning the measured callable.
3. Time only the intended region. Put output checks in the validator and reuse
   published references where available, loading them outside timing.
4. Make repeated calls independent. Use read-only inputs where possible. For
   mutating APIs, use ``benchmark.pedantic(setup=...)`` to clone inputs before
   each timed round; resetting once is insufficient for a multi-iteration round.
5. Add the case and its parameters to ``tests/benchmarks/cases.py`` so timing and
   memory select the same definition. Mark costly cases ``benchmark_slow`` and
   filesystem cases ``benchmark_io`` as appropriate.
6. Include the explicit ``benchmark`` fixture in each timing test so exclusion
   works. Call ``measure(case.build())``, run the test, and inspect its saved JSON.
7. Run the case in isolated RSS mode and check its setup policy and validator.
   Keep memory instrumentation separate from timing runs.
