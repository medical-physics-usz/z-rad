from dataclasses import dataclass


@dataclass
class BatchResult:
    """Summary returned by batch workflows.

    The individual case-result objects are workflow-specific. They are expected
    to expose a ``status`` field and may expose an ``error`` field.
    """

    workflow: str
    case_results: list

    @property
    def total_count(self) -> int:
        return len(self.case_results)

    @property
    def processed_count(self) -> int:
        return sum(result.status == 'processed' for result in self.case_results)

    @property
    def skipped_count(self) -> int:
        return sum(result.status == 'skipped' for result in self.case_results)

    @property
    def failed_count(self) -> int:
        return sum(result.status == 'failed' for result in self.case_results)

    @property
    def errors(self) -> list:
        return [result for result in self.case_results if getattr(result, 'error', None)]
