class ExtractionError(Exception):
    """Permanent failure: retrying will not help (bad file, unsupported type, ...)."""
