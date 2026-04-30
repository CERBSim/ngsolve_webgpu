import time
import threading


class AnimationLoop:
    """Reusable timer loop that calls tick() at a target fps in a daemon thread."""

    def __init__(self, fps=60):
        self._fps = fps
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread = None

    @property
    def running(self):
        return self._running

    def _loop(self):
        while self._running:
            t = time.time() - self._t0
            self.tick(t)
            time.sleep(1 / self._fps)

    def tick(self, t: float):
        """Override in subclass. Called with elapsed time in seconds."""
        raise NotImplementedError
