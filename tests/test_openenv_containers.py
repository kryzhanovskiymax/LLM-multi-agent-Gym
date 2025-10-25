from __future__ import annotations

import http.client
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_SPECS = [
    (
        "coding",
        PROJECT_ROOT / "src" / "envs" / "openenv" / "multiagentbench" / "coding" / "server" / "Dockerfile",
    ),
    (
        "minecraft",
        PROJECT_ROOT / "src" / "envs" / "openenv" / "multiagentbench" / "minecraft" / "server" / "Dockerfile",
    ),
]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            time.sleep(0.5)
            continue
        finally:
            try:
                connection.close()
            except Exception:
                pass
        time.sleep(0.5)
    raise AssertionError("Container health check did not succeed within timeout.")


def _ensure_docker_running() -> None:
    try:
        subprocess.run(
            ["docker", "version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required for container integration tests.")
@pytest.mark.parametrize(("env_name", "dockerfile"), ENV_SPECS)
def test_openenv_container_builds_and_serves_health(env_name: str, dockerfile: Path) -> None:
    _ensure_docker_running()
    tag = f"llm-mab-{env_name}-{uuid.uuid4().hex[:8]}"
    container_id: str | None = None

    try:
        subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(dockerfile), "."],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        port = _find_free_port()
        run = subprocess.run(
            ["docker", "run", "-d", "-p", f"{port}:8000", tag],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        container_id = run.stdout.strip()

        _wait_for_health(port)
    finally:
        if container_id:
            subprocess.run(["docker", "stop", container_id], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["docker", "image", "rm", "-f", tag], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
