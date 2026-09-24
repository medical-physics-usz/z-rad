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

    def calculate_selected(self, context, prepared_data, selected_features):
        """Evaluate selected outputs; groups may avoid unrelated calculations.

        The extractor still filters and orders the returned feature dictionary.
        """
        return self.calculate(context, prepared_data)
