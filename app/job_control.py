"""Shared cancel / stop control for long-running generation jobs."""
from __future__ import annotations

import threading
import time
from contextvars import ContextVar, Token
from typing import Callable


class CancelledError(RuntimeError):
    """Raised when the user (or server shutdown) stops a generation job."""


class JobControl:
    """Thread-safe stop signal for one pipeline/Comfy job."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.prompt_id: str | None = None
        self.project_id: str | None = None
        self.thread: threading.Thread | None = None
        self._interrupt_nudge: threading.Thread | None = None
        self.job_key: str | None = None

    def reset(self) -> None:
        self._stop.clear()
        with self._lock:
            self.prompt_id = None
            # keep job_key / project_id if already set by the worker

    def request_stop(self) -> None:
        self._stop.set()

    def is_cancelled(self) -> bool:
        return self._stop.is_set()

    def check(self) -> None:
        if self._stop.is_set():
            raise CancelledError("Generation cancelled by user")

    def set_prompt_id(self, prompt_id: str | None) -> None:
        with self._lock:
            self.prompt_id = prompt_id

    def set_project_id(self, project_id: str | None) -> None:
        with self._lock:
            self.project_id = project_id

    def bind_thread(self, thread: threading.Thread | None) -> None:
        self.thread = thread

    def worker_alive(self) -> bool:
        t = self.thread
        return bool(t and t.is_alive())

    def start_interrupt_nudge(
        self,
        interrupt_fn: Callable[[], None],
        *,
        interval_sec: float = 2.0,
        max_sec: float = 120.0,
    ) -> None:
        """While cancelled and the worker is alive, keep poking Comfy interrupt."""
        if self._interrupt_nudge and self._interrupt_nudge.is_alive():
            return

        def loop() -> None:
            deadline = time.time() + max_sec
            while self._stop.is_set() and self.worker_alive() and time.time() < deadline:
                try:
                    interrupt_fn()
                except Exception:
                    pass
                end = time.time() + interval_sec
                while time.time() < end:
                    if not self._stop.is_set() or not self.worker_alive():
                        return
                    time.sleep(0.15)

        self._interrupt_nudge = threading.Thread(
            target=loop, name="h3-stop-nudge", daemon=True
        )
        self._interrupt_nudge.start()


# Fallback process-wide controller (used when no worker context is bound).
_default_control = JobControl()
_default_control.job_key = "default"

_ctx_control: ContextVar[JobControl | None] = ContextVar(
    "h3_active_job_control", default=None
)


def get_job_control() -> JobControl:
    """Active job control for this worker thread/context, or the process default."""
    c = _ctx_control.get()
    return c if c is not None else _default_control


def bind_job_control(control: JobControl | None) -> Token:
    return _ctx_control.set(control)


def reset_job_control(token: Token) -> None:
    _ctx_control.reset(token)


class _JobControlProxy:
    """
    Module-level `job_control` delegates to the context-bound control so nested
    code (character sheets, LLM router) cancels the correct parallel job.
    """

    def check(self) -> None:
        get_job_control().check()

    def is_cancelled(self) -> bool:
        return get_job_control().is_cancelled()

    def request_stop(self) -> None:
        get_job_control().request_stop()

    def reset(self) -> None:
        get_job_control().reset()

    def set_prompt_id(self, prompt_id: str | None) -> None:
        get_job_control().set_prompt_id(prompt_id)

    def set_project_id(self, project_id: str | None) -> None:
        get_job_control().set_project_id(project_id)

    def bind_thread(self, thread: threading.Thread | None) -> None:
        get_job_control().bind_thread(thread)

    def worker_alive(self) -> bool:
        return get_job_control().worker_alive()

    def start_interrupt_nudge(
        self,
        interrupt_fn: Callable[[], None],
        *,
        interval_sec: float = 2.0,
        max_sec: float = 120.0,
    ) -> None:
        get_job_control().start_interrupt_nudge(
            interrupt_fn, interval_sec=interval_sec, max_sec=max_sec
        )

    @property
    def prompt_id(self) -> str | None:
        return get_job_control().prompt_id

    @prompt_id.setter
    def prompt_id(self, value: str | None) -> None:
        get_job_control().set_prompt_id(value)

    @property
    def project_id(self) -> str | None:
        return get_job_control().project_id

    @project_id.setter
    def project_id(self, value: str | None) -> None:
        get_job_control().set_project_id(value)

    @property
    def thread(self) -> threading.Thread | None:
        return get_job_control().thread

    @thread.setter
    def thread(self, value: threading.Thread | None) -> None:
        get_job_control().bind_thread(value)


# Compatible name for existing imports
job_control = _JobControlProxy()


# Optional callable invoked when stop is requested (e.g. interrupt ComfyUI).
_on_stop_hooks: list[Callable[[], None]] = []


def register_stop_hook(fn: Callable[[], None]) -> None:
    _on_stop_hooks.append(fn)


def fire_stop_hooks() -> None:
    for fn in list(_on_stop_hooks):
        try:
            fn()
        except Exception:
            pass
