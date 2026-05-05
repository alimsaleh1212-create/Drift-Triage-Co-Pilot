"""BaseTool ABC: the only entry point the LangGraph node calls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class ToolResult(BaseModel, Generic[OutputT]):
    """Wrapper returned by safe_run — never raises."""

    ok: bool
    result: OutputT | None = None
    error: str | None = None
    retryable: bool = False


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract base for all agent tools.

    Subclasses set class variables `name`, `input_schema`, `output_schema`
    and implement `run()`. The LangGraph node calls `safe_run()` only.
    """

    name: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, args: InputT) -> OutputT:
        """Execute the tool with validated args.

        Raises:
            Any exception — safe_run() will catch and wrap it.
        """
        ...

    async def safe_run(self, raw_args: dict[str, object]) -> ToolResult[OutputT]:
        """Validate raw_args, call run(), wrap exceptions as ToolResult.

        Never raises. Validation failures are returned as ToolResult(ok=False)
        so the LLM can retry with corrected args.

        Args:
            raw_args: Dict from LLM tool-call JSON.

        Returns:
            ToolResult with ok=True and result on success, ok=False and error otherwise.
        """
        import httpx
        from pydantic import ValidationError

        from core.logging import get_logger

        log = get_logger(__name__)

        try:
            args: InputT = self.input_schema.model_validate(raw_args)  # type: ignore[assignment]
        except ValidationError as exc:
            log.warning("tool.validation_error", tool=self.name, error=str(exc))
            return ToolResult(ok=False, error=f"invalid args: {exc}", retryable=False)  # type: ignore[return-value]

        try:
            result = await self.run(args)
            return ToolResult(ok=True, result=result)  # type: ignore[return-value]
        except httpx.HTTPStatusError as exc:
            return ToolResult(  # type: ignore[return-value]
                ok=False,
                error=f"upstream {exc.response.status_code}",
                retryable=False,
            )
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            return ToolResult(ok=False, error=f"unreachable: {exc}", retryable=True)  # type: ignore[return-value]
        except Exception as exc:
            log.exception("tool.unexpected_error", tool=self.name)
            return ToolResult(ok=False, error=str(exc), retryable=False)  # type: ignore[return-value]
