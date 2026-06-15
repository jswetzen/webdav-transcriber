from __future__ import annotations

import sys

from .config import Config
from .webdav import WebDAVClient


def main() -> None:
    """Liveness probe for a single WebDAV watcher container.

    Exercises auth + reachability of the configured share by listing the watch
    path: exits 0 on success, 1 on any failure (401, host unreachable, missing
    path). Designed to back a podman ``--health-cmd`` so a persistently broken
    share flips its unit to a degraded state the Pulse agent can alert on.

    Single-watcher by construction: each poller container runs one share's config,
    so this probes only that share. With one container per watcher, a bad
    credential trips only its own unit and never the others.
    """
    try:
        config = Config()
    except Exception as exc:
        print(f"healthcheck: invalid config: {exc}", file=sys.stderr)
        sys.exit(1)

    if not config.webdav_url:
        # No share configured (e.g. a server-only deployment): nothing to probe.
        print("healthcheck: no WEBDAV_URL configured; skipping", file=sys.stderr)
        sys.exit(0)

    try:
        WebDAVClient(config).check_connection()
    except Exception as exc:
        print(
            f"healthcheck: WebDAV unreachable or unauthorized "
            f"({config.webdav_url} {config.webdav_watch_path}): {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
