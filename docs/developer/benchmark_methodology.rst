Benchmark methodology and maintenance
=====================================

Read this page when interpreting measurement boundaries or adding workloads.
For commands to run and compare results, start with :doc:`benchmarking`.
For RSS and allocation measurements, see :doc:`memory_profiling`.

Measured operations
-------------------

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
-----------------------------

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
-----------------------

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
-------------------------------------

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
