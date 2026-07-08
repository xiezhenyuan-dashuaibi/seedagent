from __future__ import annotations

from collections.abc import Generator
from typing import Any

from smolagents import OpenAIModel
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCallStreamDelta,
    agglomerate_stream_deltas,
)
from smolagents.monitoring import TokenUsage

try:
    from .context_logger import ModelInputLogger
    from .runtime_events import RuntimeEventRouter
except ImportError:
    from context_logger import ModelInputLogger
    from runtime_events import RuntimeEventRouter


COMMENTARY_OPEN = "<commentary>"
COMMENTARY_CLOSE = "</commentary>"
PYTHON_RUN_OPEN = "<python_run>"
PYTHON_RUN_CLOSE = "</python_run>"
SHELL_RUN_OPEN = "<shell_run>"
SHELL_RUN_CLOSE = "</shell_run>"
FINAL_OPEN = "<final>"
FINAL_CLOSE = "</final>"


RUN_BLOCK_TAGS = (
    (PYTHON_RUN_OPEN, PYTHON_RUN_CLOSE),
    (SHELL_RUN_OPEN, SHELL_RUN_CLOSE),
)


def _truncate_after_first_run_block(previous_text: str, delta_text: str) -> tuple[str, bool]:
    combined_text = previous_text + delta_text

    earliest_stop_index: int | None = None
    for opening_tag, closing_tag in RUN_BLOCK_TAGS:
        run_open_index = combined_text.find(opening_tag)
        if run_open_index == -1:
            continue
        run_close_index = combined_text.find(closing_tag, run_open_index + len(opening_tag))
        if run_close_index == -1:
            continue
        stop_index = run_close_index + len(closing_tag)
        if earliest_stop_index is None or stop_index < earliest_stop_index:
            earliest_stop_index = stop_index

    if earliest_stop_index is None:
        return delta_text, False

    keep_length = max(0, min(len(delta_text), earliest_stop_index - len(previous_text)))
    return delta_text[:keep_length], True


def _starts_with_routing_tag(text: str) -> bool | None:
    stripped_text = text.lstrip()
    if not stripped_text:
        return None
    return stripped_text.startswith("<")


class TaggedTextStreamer:
    """Incrementally extracts text inside a pair of routing tags from streamed chunks."""

    def __init__(
        self,
        router: RuntimeEventRouter,
        *,
        opening_tag: str,
        closing_tag: str,
        delta_event: str,
        finished_event: str,
    ) -> None:
        self.router = router
        self.opening_tag = opening_tag
        self.closing_tag = closing_tag
        self.delta_event = delta_event
        self.finished_event = finished_event
        self._buffer = ""
        self._inside_tag = False

    def feed(self, text: str) -> None:
        if not text:
            return
        self._buffer += text
        self._drain()

    def finish(self) -> None:
        if self._inside_tag and self._buffer:
            self.router.emit(self.delta_event, text=self._buffer)
            self._buffer = ""
        if self._inside_tag:
            self.router.emit(self.finished_event)
        self._inside_tag = False
        self._buffer = ""

    def _drain(self) -> None:
        while self._buffer:
            if self._inside_tag:
                close_index = self._buffer.find(self.closing_tag)
                if close_index == -1:
                    keep = len(self.closing_tag) - 1
                    emit_upto = max(0, len(self._buffer) - keep)
                    if emit_upto:
                        self.router.emit(self.delta_event, text=self._buffer[:emit_upto])
                        self._buffer = self._buffer[emit_upto:]
                    return

                if close_index:
                    self.router.emit(self.delta_event, text=self._buffer[:close_index])
                self._buffer = self._buffer[close_index + len(self.closing_tag) :]
                self._inside_tag = False
                self.router.emit(self.finished_event)
                continue

            open_index = self._buffer.find(self.opening_tag)
            if open_index == -1:
                keep = len(self.opening_tag) - 1
                if len(self._buffer) > keep:
                    self._buffer = self._buffer[-keep:]
                return

            self._buffer = self._buffer[open_index + len(self.opening_tag) :]
            self._inside_tag = True


class SeedStreamingOpenAIModel(OpenAIModel):
    """OpenAIModel that streams SeedAgent routing events while returning a normal ChatMessage."""

    def __init__(
        self,
        *args: Any,
        router: RuntimeEventRouter,
        model_input_logger: ModelInputLogger | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.router = router
        self.model_input_logger = model_input_logger

    def _raw_generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        stream = self.retryer(
            self.client.chat.completions.create,
            **completion_kwargs,
            stream=True,
            stream_options={"include_usage": True},
        )
        try:
            for event in stream:
                if getattr(event, "usage", None):
                    yield ChatMessageStreamDelta(
                        content="",
                        token_usage=TokenUsage(
                            input_tokens=event.usage.prompt_tokens,
                            output_tokens=event.usage.completion_tokens,
                        ),
                    )
                if event.choices:
                    choice = event.choices[0]
                    if choice.delta:
                        yield ChatMessageStreamDelta(
                            content=choice.delta.content,
                            tool_calls=[
                                ChatMessageToolCallStreamDelta(
                                    index=delta.index,
                                    id=delta.id,
                                    type=delta.type,
                                    function=delta.function,
                                )
                                for delta in choice.delta.tool_calls
                            ]
                            if choice.delta.tool_calls
                            else None,
                        )
                    elif not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        if self.model_input_logger is not None:
            self.model_input_logger.log_model_input(
                messages,
                model_id=getattr(self, "model_id", None),
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
            )
        self.router.emit("model_output_started")
        stream_deltas: list[ChatMessageStreamDelta] = []
        commentary = TaggedTextStreamer(
            self.router,
            opening_tag=COMMENTARY_OPEN,
            closing_tag=COMMENTARY_CLOSE,
            delta_event="commentary_delta",
            finished_event="commentary_finished",
        )
        final = TaggedTextStreamer(
            self.router,
            opening_tag=FINAL_OPEN,
            closing_tag=FINAL_CLOSE,
            delta_event="final_delta",
            finished_event="final_finished",
        )
        try:
            generated_text = ""
            has_valid_routing_start: bool | None = None
            raw_stream = self._raw_generate_stream(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
            should_close_stream = False
            for delta in raw_stream:
                should_stop = False
                if delta.content:
                    kept_content, should_stop = _truncate_after_first_run_block(generated_text, delta.content)
                    if kept_content != delta.content:
                        delta = ChatMessageStreamDelta(
                            content=kept_content,
                            tool_calls=delta.tool_calls,
                            token_usage=delta.token_usage,
                        )
                    if has_valid_routing_start is None:
                        has_valid_routing_start = _starts_with_routing_tag(generated_text + kept_content)
                        if has_valid_routing_start is False:
                            generated_text += kept_content
                            stream_deltas.append(delta)
                            should_close_stream = True
                            break
                    generated_text += kept_content

                stream_deltas.append(delta)
                if delta.content and has_valid_routing_start:
                    commentary.feed(delta.content)
                    final.feed(delta.content)
                if should_stop:
                    should_close_stream = True
                    break
            if should_close_stream:
                raw_stream.close()
        finally:
            commentary.finish()
            final.finish()
            self.router.emit("model_output_finished")

        return agglomerate_stream_deltas(stream_deltas)

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageStreamDelta]:
        for delta in super().generate_stream(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        ):
            yield delta


__all__ = [
    "COMMENTARY_CLOSE",
    "COMMENTARY_OPEN",
    "FINAL_CLOSE",
    "FINAL_OPEN",
    "PYTHON_RUN_CLOSE",
    "PYTHON_RUN_OPEN",
    "SHELL_RUN_CLOSE",
    "SHELL_RUN_OPEN",
    "TaggedTextStreamer",
    "_truncate_after_first_run_block",
    "_starts_with_routing_tag",
    "SeedStreamingOpenAIModel",
    "RoutedStreamingOpenAIModel",
]


RoutedStreamingOpenAIModel = SeedStreamingOpenAIModel
