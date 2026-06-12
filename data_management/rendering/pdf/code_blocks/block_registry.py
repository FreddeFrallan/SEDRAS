import inspect
from typing import Callable, Dict, Any


class BlockLibrary:
    def __init__(self):
        self.registry: Dict[str, Callable] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def register_block(self, name: str = None):
        """Decorator to register a function as a PDF block."""

        def decorator(func: Callable):
            block_name = name or func.__name__
            self.registry[block_name] = func

            # Extract docstring and signature for the LLM prompt
            sig = inspect.signature(func)
            # Filter out mandatory internal args so the LLM doesn't see them
            internal_args = {'theme', 'tracker', 'story', 'styles', 'llm'}
            external_params = [
                p for p in sig.parameters.values()
                if p.name not in internal_args
            ]

            self.metadata[block_name] = {
                "description": func.__doc__ or "No description provided.",
                "signature": f"{block_name}({', '.join(str(p) for p in external_params)})"
            }
            return func

        return decorator

    def get_prompt_snippet(self) -> str:
        """Generates a documentation string for the LLM prompt."""
        lines = ["## Available UI Blocks\n"]
        assert len(self.metadata.items()) > 0, "No blocks registered in the library."
        for name, meta in self.metadata.items():
            lines.append(f"### `{meta['signature']}`")
            lines.append(f"{meta['description']}\n")
        return "\n".join(lines)


# Global instance
library = BlockLibrary()
register_block = library.register_block