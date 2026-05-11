class Pipeline:
    """Run named preprocessing steps in sequence."""

    def __init__(self, steps):
        self.steps = list(steps)
        self._validate_steps()

    def get_params(self):
        """Return pipeline step parameters mapped by step name."""
        params = {}
        for name, step in self.steps:
            params[name] = step.get_params() if hasattr(step, "get_params") else {}
        return params

    def apply(self, data):
        """Apply all steps and return the transformed data."""
        result = data
        for _name, step in self.steps:
            result = step.apply(result)
        return result

    def _validate_steps(self):
        for step_def in self.steps:
            if not isinstance(step_def, tuple) or len(step_def) != 2:
                raise ValueError("Pipeline steps must be (name, step) tuples.")
            name, step = step_def
            if not isinstance(name, str) or not name:
                raise ValueError("Pipeline step names must be non-empty strings.")
            if not hasattr(step, "apply"):
                raise TypeError(f"Pipeline step '{name}' must expose an apply method.")
