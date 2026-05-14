class BaseFeatureGroup:
    """Minimal contract for radiomics feature-family groups."""

    family = None
    requirements = frozenset()

    def supports(self, context):
        return True

    def default_enabled(self, context):
        return self.supports(context)

    def output_names(self, context):
        raise NotImplementedError

    def feature_aliases(self, context):
        output_names = self.output_names(context)
        return {name: name for name in output_names}

    def calculate(self, context, prepared_data):
        raise NotImplementedError
