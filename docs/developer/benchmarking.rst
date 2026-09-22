Performance benchmarking
========================

The benchmarks in ``tests/benchmarks`` measure Z-Rad execution time and memory
use. Use individual operation benchmarks to identify where performance changed,
and IBSI workflows to assess the effect on complete processing workflows.
Setup and validation run outside the timed region. For memory measurements,
see :doc:`memory_profiling`; for workload boundaries and adding benchmarks,
see :doc:`benchmark_methodology`.

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

.. list-table:: Choose a comparison workflow
   :header-rows: 1
   :widths: 40 60

   * - What you want to measure
     - Approach
   * - Uncommitted production changes
     - Run ``run.py`` in the working tree and save the result.
   * - Two committed revisions
     - Use ``compare_revisions.py`` to install and measure each revision.
   * - Previously saved measurements
     - Use ``compare_results.py`` with explicit reference files.


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

Results from CI
---------------

The performance workflow publishes comparison summaries and raw artifacts;
see :doc:`ci` for triggers and artifact names. On pull requests, the candidate
is GitHub's tested merge commit. The master reference is resolved at checkout;
on pushes to master, candidate and reference may be identical, providing an
observation of measurement noise. The latest published stable release is
resolved to a fixed SHA for that job. These runs do not replace your saved
accepted references.

GitHub-hosted hardware and background load vary. Treat the measurements as
performance indicators and investigate changes locally. Candidate measurement
or validation failures fail the job; timing and memory changes have no
configured regression threshold.

Advanced: named saves
---------------------

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

For direct pytest invocation and its thread-control requirements, see
:doc:`benchmark_methodology`.
