"""Declarative launch configuration for single-node vLLM serving."""


class ConfigError(ValueError):
    """The requested configuration has no unambiguous supported launch plan."""
