from typing import Any, Optional


class InvalidTemplateError(Exception):
    """Raised when an LLM returns an invalid or non-parseable template.

    Attributes:
        llm_response: Optional raw response returned by the LLM that triggered
            the error. This is useful for smarter retries where we want to feed
            the previous response back into the model.
    """

    def __init__(self, message: str, *, llm_response: Optional[Any] = None):
        super().__init__(message)
        self.llm_response = llm_response
