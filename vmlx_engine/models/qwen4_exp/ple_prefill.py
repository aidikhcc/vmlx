"""Request-owned, host-only lookahead for known prompt chunks.

The scheduler owns this scope. Nothing is stored in the model's native cache;
the ordinary PLE lookup still validates exact row identity and updates history.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PLEChunkReadAhead:
    def __init__(self, layers):
        self.layers = layers
        self.pending = {}
        self.stats = {"prepared": 0, "matched": 0, "discarded": 0}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _drain(self):
        tickets, self.pending = self.pending, {}
        for ticket in tickets.values():
            ticket.close()

    def close(self):
        if not self.closed:
            self.closed = True
            self._drain()
            if self.stats["prepared"]:
                logger.info("Qwen PLE chunk lookahead: %s drained=true", self.stats)

    def prepare(self, input_ids, cache):
        """Read future known IDs against the *already advanced* native history."""
        if self.closed:
            raise RuntimeError("PLE chunk lookahead is closed")
        self._drain()
        if input_ids.shape[0] != 1 or input_ids.shape[1] <= 1:
            return
        try:
            for index, (layer, current) in enumerate(zip(self.layers, cache)):
                if layer.ple is not None:
                    ticket = layer.ple.prepare_read(input_ids, current, lookahead=True)
                    if ticket is not None:
                        self.pending[index] = ticket
                        self.stats["prepared"] += 1
        except BaseException:
            self._drain()
            raise

    def take(self, input_ids, cache):
        """A changed chunk/history falls back; never consume a predicted prefix."""
        if self.closed:
            raise RuntimeError("PLE chunk lookahead is closed")
        prepared = {}
        for index, ticket in self.pending.items():
            if ticket.future is None:
                continue
            ple = self.layers[index].ple
            rows = ple.read_rows(input_ids, cache[index])
            if np.array_equal(rows, ticket.rows):
                prepared[index] = ticket
                self.stats["matched"] += 1
            else:
                ticket.close()
                self.stats["discarded"] += 1
        # Retain ownership until the next prepare/close, including exceptions
        # before the model consumes the borrowed tickets. close is idempotent.
        return prepared
