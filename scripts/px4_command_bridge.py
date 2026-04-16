#!/usr/bin/env python3
"""Forward newline-delimited TCP commands to the PX4 shell FIFO."""

from __future__ import annotations

import argparse
import logging
import socketserver
import threading


class CommandForwarder:
    """Serialize writes into the PX4 shell pipe."""

    def __init__(self, pipe_path: str) -> None:
        self._pipe = open(pipe_path, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def write_line(self, command: str) -> None:
        command = command.strip()
        if not command:
            return
        with self._lock:
            self._pipe.write(command + "\n")
            self._pipe.flush()

    def close(self) -> None:
        self._pipe.close()


class CommandBridgeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[socketserver.StreamRequestHandler],
        forwarder: CommandForwarder,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.forwarder = forwarder


class CommandHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        logging.info("client connected: %s", peer)
        try:
            for raw_line in self.rfile:
                try:
                    command = raw_line.decode("utf-8").strip()
                except UnicodeDecodeError:
                    logging.warning("dropping non-UTF-8 command from %s", peer)
                    continue

                if not command:
                    continue

                logging.info("forwarding command: %s", command)
                self.server.forwarder.write_line(command)
        finally:
            logging.info("client disconnected: %s", peer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipe", required=True, help="Path to the PX4 shell FIFO")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address")
    parser.add_argument(
        "--port",
        type=int,
        default=14600,
        help="Listen port for incoming shell commands",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[px4-command-bridge] %(message)s",
    )

    forwarder = CommandForwarder(args.pipe)
    with CommandBridgeServer(
        (args.host, args.port),
        CommandHandler,
        forwarder,
    ) as server:
        logging.info(
            "listening on %s:%d -> %s", args.host, args.port, args.pipe
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            forwarder.close()


if __name__ == "__main__":
    main()
