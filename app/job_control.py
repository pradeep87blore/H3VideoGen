"""Shared cancel / stop control for long-running generation jobs."""
from __future__ import annotations

import threading
import time
from typing import Callable


class CancelledError(RuntimeError):
    """Raised when the user (or server shutdown) stops a generation job."""


class JobControl:
    """Thread-safe stop signal for the active pipeline/Comfy job."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.prompt_id: str | None = None
        self.project_id: str | None = None
        self.thread: threading.Thread | None = None
        self._interrupt_nudge: threading.Thread | None = None

    def reset(self) -> None:
        self._stop.clear()
        with self._lock:
            self.prompt_id = None
            self.project_id = None

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


# Process-wide controller (one heavy job at a time).
job_control = JobControl()


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
