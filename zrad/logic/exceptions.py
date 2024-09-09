"""Custom warnings and errors used across Z-Rad."""


class InvalidInputParametersError(ValueError):
    """Custom exception to indicate invalid input parameters."""


class DataStructureWarning(UserWarning):
    """Custom exception to indicate problems with input data structure."""


class DataStructureError(Exception):
    """Custom exception to indicate invalid input data structure."""
