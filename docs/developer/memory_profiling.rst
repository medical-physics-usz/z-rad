Memory profiling
================

Use peak process memory to identify expensive workloads, then allocation
profiling to investigate where memory is allocated. Run commands from the
repository root in the :doc:`development_environment`.
Timing and memory share the suites described in :doc:`benchmarking`.

Peak process memory
-------------------

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

With multiple repetitions, the RSS summary shows median peaks and the observed
range for each workload. Use these to assess variability before comparing runs.

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
--------------------

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
-------------------

Memory JSON uses ``schema_version: 1`` and a ``measurements`` list. Each entry
includes a workload ID, revision and dirty flag, PID, timestamp, Python/platform,
workload/environment metadata, ``memory_methodology_version`` (1 for setup before
operation), and mode-specific measurements. Repetitions remain separate, and
existing output paths are refused.

Join ``measurements[].workload_id`` with timing ``extra_info.workload_id`` to
inspect both metrics for a case. Measure run-to-run variability on your platform
before setting memory failure thresholds.


To regenerate a Markdown summary from an existing RSS report, run::

    python tests/benchmarks/summarize_memory.py reports/benchmarks/rss-standard-001.json

See :doc:`benchmark_methodology` for input preparation and thread controls,
and :doc:`ci` for scheduled memory runs and their artifacts.
