Continuous integration
======================

GitHub Actions runs the workflows below. Open a workflow run to inspect a failed
step's log or download its artifacts. For commands to reproduce checks locally,
see :doc:`testing`, :doc:`code_quality`, and :doc:`building_docs`.

Workflows and triggers
----------------------

Workflow files are in ``.github/workflows/``. In this table, PR means a pull
request targeting ``master`` and push means a push to ``master``.

.. list-table:: Automated workflows
   :header-rows: 1
   :widths: 25 25 50

   * - Workflow file
     - Trigger
     - Purpose and result
   * - ``test.yml``
     - PR and push
     - Unit and integration tests on Python 3.11, 3.12, 3.13, and 3.14;
       coverage and IBSI validation artifacts.
   * - ``python-lint.yml``
     - PR and push
     - Ruff formatting and lint checks on Python 3.11.
   * - ``lint.yml``
     - PR and push
     - Super-Linter checks on changed files.
   * - ``docs.yml``
     - Push and manual dispatch
     - Strict Sphinx build on Python 3.12, followed by GitHub Pages deployment.
   * - ``benchmark.yml``
     - PR, push, weekly schedule, and manual dispatch
     - Timing comparisons; scheduled and exhaustive manual runs also measure RSS.
   * - ``publish.yml``
     - GitHub release published
     - Build and upload Python distributions to PyPI.
   * - ``release-executables.yml``
     - GitHub release published
     - Build and attach desktop application assets.

Documentation is not built on pull requests. Run the local Sphinx build for
documentation changes even when the other PR checks pass. The documentation
workflow's deploy job follows a successful build, including on manual runs.

Inspect test and coverage results
---------------------------------

The test workflow runs the unit suite, then appends integration-test coverage.
It uploads ``coverage-report-python-*`` HTML artifacts and
``ibsi-results-python-*`` validation reports for each Python version. Validation
reports include failures and skips when integration results are available.
See :doc:`ibsi_validation` to reproduce and interpret them.

Coverage has no configured minimum percentage. Review coverage for the changed
code along with the test results; a passing workflow is not a coverage target.

Inspect performance results
---------------------------

Performance jobs run on Ubuntu 24.04 with Python 3.12. PRs and pushes select the
``standard`` suite. Manual dispatch selects a suite, and the weekly schedule
selects ``exhaustive``. RSS memory runs occur on the weekly schedule and on
manual runs selecting ``exhaustive``, with one sample per workload.

The job summaries show timing comparisons or grouped peak/setup-peak RSS tables.
Artifacts are retained for 30 days:

* ``performance-<run-id>-<attempt>`` contains available timing JSON, comparison
  output, installation/test logs, dependency lists, and revision/status metadata.
* ``performance-memory-<run-id>-<attempt>`` contains RSS JSON and Markdown reports.

Candidate measurement or validation failures fail the timing job. There is no
configured threshold that fails a run simply because time or memory increased.
Missing or incompatible references are reported separately; inspect the status
and logs before interpreting a partial comparison. See :doc:`benchmarking` for
reference selection and :doc:`memory_profiling` for memory metrics.

Release jobs
------------

Publishing a GitHub release triggers both package publication and executable
builds. These are separate workflows, so verify both results. See
:doc:`release_process` for preparation, version conventions, and expected assets.
