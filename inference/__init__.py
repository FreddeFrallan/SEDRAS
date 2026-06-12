from inference.enums import LLMBackend, LLMModel

__all__ = [
    "CLASSIFIER_SYSTEM_PROMPT",
    "LLMBackend",
    "LLMModel",
    "LLMWrapper",
    "ChatMessage",
    "GeminiWrapper",
    "LiteLLMChatWrapper",
    "OpenAIChatWrapper",
    "DeepSeekWrapper",
    "RemoteHTTPWrapper",
    "CustomHTTPBackendWrapper",
    "get_llm_wrapper",
    "render_classifier_user_prompt",
]


def __getattr__(name):
    """
    Lazily import heavy wrappers only when requested.

    This keeps lightweight enums available without requiring optional
    dependencies (e.g., docstring_parser) during simple config imports.
    """
    if name in __all__:
        from inference.model_wrappers.llm_wrapper import (
            CLASSIFIER_SYSTEM_PROMPT,
            LLMWrapper,
            ChatMessage,
            GeminiWrapper,
            LiteLLMChatWrapper,
            OpenAIChatWrapper,
            DeepSeekWrapper,
            RemoteHTTPWrapper,
            CustomHTTPBackendWrapper,
            get_llm_wrapper,
            render_classifier_user_prompt,
        )

        globals().update(
            {
                "CLASSIFIER_SYSTEM_PROMPT": CLASSIFIER_SYSTEM_PROMPT,
                "LLMWrapper": LLMWrapper,
                "ChatMessage": ChatMessage,
                "GeminiWrapper": GeminiWrapper,
                "LiteLLMChatWrapper": LiteLLMChatWrapper,
                "OpenAIChatWrapper": OpenAIChatWrapper,
                "DeepSeekWrapper": DeepSeekWrapper,
                "RemoteHTTPWrapper": RemoteHTTPWrapper,
                "CustomHTTPBackendWrapper": CustomHTTPBackendWrapper,
                "get_llm_wrapper": get_llm_wrapper,
                "render_classifier_user_prompt": render_classifier_user_prompt,
            }
        )
        return globals()[name]
    raise AttributeError(f"module 'inference' has no attribute '{name}'")
