from dataclasses import dataclass


@dataclass
class BatchResult:
    """Summary returned by batch workflows.

    ``BatchResult`` provides shared aggregate reporting for preprocessing,
    filtering, and radiomics batch runs. The individual case-result objects are
    workflow-specific, but they all expose a ``status`` field and may expose an
    ``error`` field.

    Parameters
    ----------
    workflow : str
        Name of the batch workflow, such as ``"preprocessing"``,
        ``"filtering"``, or ``"radiomics"``.
    case_results : list
        Workflow-specific case-result objects.
    """

    workflow: str
    case_results: list

    @property
    def total_count(self) -> int:
        """Total number of selected cases.

        Returns
        -------
        count : int
            Number of case results stored in ``case_results``.
        """
        return len(self.case_results)

    @property
    def processed_count(self) -> int:
        """Number of cases with ``status == "processed"``.

        Returns
        -------
        count : int
            Count of successfully processed case results.
        """
        return sum(result.status == 'processed' for result in self.case_results)

    @property
    def skipped_count(self) -> int:
        """Number of cases with ``status == "skipped"``.

        Returns
        -------
        count : int
            Count of skipped case results.
        """
        return sum(result.status == 'skipped' for result in self.case_results)

    @property
    def failed_count(self) -> int:
        """Number of cases with ``status == "failed"``.

        Returns
        -------
        count : int
            Count of failed case results.
        """
        return sum(result.status == 'failed' for result in self.case_results)

    @property
    def errors(self) -> list:
        """Case results that include an error message.

        Returns
        -------
        errors : list
            Case-result objects where ``error`` is not empty.
        """
        return [result for result in self.case_results if getattr(result, 'error', None)]
