class ExpertPackError(RuntimeError):
    """Base class for deterministic compiler failures."""


class SourceFormatError(ExpertPackError):
    """The source checkpoint is malformed or unsupported."""


class AdapterError(ExpertPackError):
    """The selected architecture adapter cannot identify the checkpoint."""


class ResumeError(ExpertPackError):
    """A partial conversion cannot safely be resumed."""


class ValidationError(ExpertPackError):
    """An Expert Pack container failed independent validation."""

