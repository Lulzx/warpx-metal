#!/usr/bin/env python3
"""Run WarpX in checkpointed process generations for Metal fault recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MARKER_NAME = ".warpx-metal-complete.json"
TIMEOUT_PATTERNS = (
    "metal_queue:",
    "metal_node_event:",
    "timed out waiting for shared-event",
    "command buffer remained",
)
RESERVED_ARGUMENTS = (
    "max_step",
    "amr.restart",
    "diagnostics.diags_names",
)
RECOVERABLE_RETURN_CODES = (-signal.SIGTERM, -signal.SIGKILL)


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path


@dataclass(frozen=True)
class AttemptResult:
    returncode: int
    watchdog_fired: bool
    metal_timeout: bool


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def assignment_key(argument: str) -> str | None:
    if "=" not in argument:
        return None
    return argument.split("=", 1)[0].strip()


def strip_reserved_arguments(arguments: Iterable[str]) -> list[str]:
    return [
        argument
        for argument in arguments
        if argument != "--" and assignment_key(argument) not in RESERVED_ARGUMENTS
    ]


def _logical_lines(path: Path) -> Iterable[str]:
    pending = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        pending += (" " if pending else "") + line.rstrip("\\").strip()
        if line.endswith("\\"):
            continue
        yield pending
        pending = ""
    if pending:
        yield pending


def read_diagnostic_names(path: Path, visited: set[Path] | None = None) -> list[str]:
    """Read the last effective diagnostics.diags_names assignment.

    AMReX input decks can include other files with ``FILE = path``. This small
    parser follows those includes in encounter order, which is sufficient for
    preserving existing diagnostics when the supervisor adds its checkpoint.
    """

    resolved = path.resolve()
    visited = visited or set()
    if resolved in visited:
        return []
    visited.add(resolved)

    names: list[str] = []
    for line in _logical_lines(resolved):
        include_match = re.fullmatch(r"FILE\s*=\s*[\"']?([^\"']+?)[\"']?", line)
        if include_match:
            include_path = Path(include_match.group(1).strip())
            if not include_path.is_absolute():
                include_path = resolved.parent / include_path
            if include_path.is_file():
                included_names = read_diagnostic_names(include_path, visited)
                if included_names:
                    names = included_names
            continue

        assignment = re.fullmatch(
            r"diagnostics\.diags_names\s*=\s*(.*)", line
        )
        if assignment:
            names = assignment.group(1).split()
    return names


def read_parameter(
    path: Path,
    key: str,
    visited: set[Path] | None = None,
) -> str | None:
    resolved = path.resolve()
    visited = visited or set()
    if resolved in visited:
        return None
    visited.add(resolved)

    value: str | None = None
    for line in _logical_lines(resolved):
        include_match = re.fullmatch(r"FILE\s*=\s*[\"']?([^\"']+?)[\"']?", line)
        if include_match:
            include_path = Path(include_match.group(1).strip())
            if not include_path.is_absolute():
                include_path = resolved.parent / include_path
            if include_path.is_file():
                included_value = read_parameter(include_path, key, visited)
                if included_value is not None:
                    value = included_value
            continue

        assignment = re.fullmatch(rf"{re.escape(key)}\s*=\s*(.*)", line)
        if assignment:
            value = assignment.group(1).strip()
    return value


def checkpoint_path(prefix: Path, step: int, digits: int) -> Path:
    return prefix.parent / f"{prefix.name}{step:0{digits}d}"


def read_verified_checkpoints(prefix: Path, max_step: int) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    if not prefix.parent.is_dir():
        return checkpoints

    pattern = re.compile(rf"^{re.escape(prefix.name)}(\d+)$")
    for candidate in prefix.parent.iterdir():
        match = pattern.match(candidate.name)
        marker = candidate / MARKER_NAME
        if not match or not marker.is_file():
            continue
        try:
            metadata = json.loads(marker.read_text(encoding="utf-8"))
            step = int(metadata["step"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if step == int(match.group(1)) and step <= max_step:
            checkpoints.append(Checkpoint(step=step, path=candidate.resolve()))
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.step)


def verify_checkpoint(path: Path) -> None:
    required = (path / "WarpXHeader", path / "Level_0")
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(
            f"checkpoint {path} is incomplete; missing: {', '.join(missing)}"
        )


def mark_checkpoint(
    path: Path,
    step: int,
    previous: Path | None,
    command: list[str],
) -> None:
    verify_checkpoint(path)
    metadata = {
        "step": step,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "previous_checkpoint": str(previous) if previous else None,
        "command": command,
    }
    temporary = path / f"{MARKER_NAME}.tmp"
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path / MARKER_NAME)


def quarantine_checkpoint(path: Path, state_dir: Path) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"refusing to move checkpoint symlink: {path}")
    quarantine_dir = state_dir / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / f"{path.name}.{time.time_ns()}"
    try:
        return Path(shutil.move(str(path), str(destination)))
    except OSError as error:
        raise RuntimeError(
            f"could not quarantine incomplete checkpoint {path}: {error}"
        ) from error


def terminate_process(process: subprocess.Popen[bytes], grace_seconds: int = 10) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_attempt(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    stall_timeout_seconds: int,
    work_dir: Path,
) -> AttemptResult:
    watchdog_fired = False
    last_size = -1
    last_progress = time.monotonic()

    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=work_dir,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                time.sleep(1)
                try:
                    size = log_path.stat().st_size
                except OSError:
                    size = last_size
                if size != last_size:
                    last_size = size
                    last_progress = time.monotonic()
                elif (
                    stall_timeout_seconds > 0
                    and time.monotonic() - last_progress >= stall_timeout_seconds
                ):
                    watchdog_fired = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            terminate_process(process)
            raise

    output = log_path.read_text(encoding="utf-8", errors="replace")
    metal_timeout = "timed out" in output and any(
        pattern in output for pattern in TIMEOUT_PATTERNS
    )
    return AttemptResult(
        returncode=process.returncode,
        watchdog_fired=watchdog_fired,
        metal_timeout=metal_timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run WarpX in short process generations. Each successful generation "
            "commits a verified checkpoint; Metal timeouts restart from the last "
            "committed checkpoint in a fresh process."
        )
    )
    parser.add_argument("--max-step", required=True, type=positive_int)
    parser.add_argument("--chunk-steps", type=positive_int, default=100)
    parser.add_argument("--max-retries", type=nonnegative_int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=nonnegative_int, default=5)
    parser.add_argument("--command-timeout-seconds", type=positive_int, default=120)
    parser.add_argument(
        "--stall-timeout-seconds",
        type=nonnegative_int,
        default=300,
        help="kill a child after this many seconds without log progress; 0 disables",
    )
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint-prefix",
        type=Path,
        default=Path(".warpx-metal-checkpoints/chk"),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(".warpx-metal-supervisor"),
    )
    parser.add_argument("--diagnostic-name", default="metal_recovery_chk")
    parser.add_argument("--checkpoint-digits", type=positive_int, default=10)
    parser.add_argument(
        "--initial-restart",
        type=Path,
        help="trusted WarpX checkpoint to use before supervisor-owned checkpoints",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("executable", type=Path)
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "warpx_arguments",
        nargs=argparse.REMAINDER,
        help="additional WarpX assignments after an optional -- separator",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = args.work_dir.resolve()
    executable = args.executable.resolve()
    input_file = args.input_file.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SystemExit(f"executable is missing or not executable: {executable}")
    if not input_file.is_file():
        raise SystemExit(f"input file does not exist: {input_file}")

    checkpoint_prefix = args.checkpoint_prefix
    if not checkpoint_prefix.is_absolute():
        checkpoint_prefix = work_dir / checkpoint_prefix
    checkpoint_prefix = checkpoint_prefix.resolve()

    state_dir = args.state_dir
    if not state_dir.is_absolute():
        state_dir = work_dir / state_dir
    state_dir = state_dir.resolve()
    checkpoint_prefix.parent.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    diagnostic_names = read_diagnostic_names(input_file)
    for argument in args.warpx_arguments:
        if assignment_key(argument) == "diagnostics.diags_names":
            diagnostic_names = argument.split("=", 1)[1].split()
    if args.diagnostic_name not in diagnostic_names:
        diagnostic_names.append(args.diagnostic_name)

    forwarded_arguments = strip_reserved_arguments(args.warpx_arguments)
    verified = (
        []
        if args.no_resume
        else read_verified_checkpoints(checkpoint_prefix, args.max_step)
    )
    current = verified[-1].step if verified else 0
    previous = verified[-1].path if verified else None
    if not previous and args.initial_restart:
        initial_restart = args.initial_restart
        if not initial_restart.is_absolute():
            initial_restart = work_dir / initial_restart
        initial_restart = initial_restart.resolve()
        try:
            verify_checkpoint(initial_restart)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        step_match = re.search(r"(\d+)$", initial_restart.name)
        if not step_match:
            raise SystemExit(
                "cannot infer the step from --initial-restart; "
                "the checkpoint directory must end in digits"
            )
        current = int(step_match.group(1))
        previous = initial_restart
    elif not previous:
        input_restart = read_parameter(input_file, "amr.restart")
        if input_restart:
            raise SystemExit(
                "the input deck sets amr.restart; pass that checkpoint through "
                "--initial-restart so the supervisor can track its step"
            )

    print("=== WarpX Metal resilient runner ===")
    print(f"Executable:        {executable}")
    print(f"Input:             {input_file}")
    print(f"Work directory:    {work_dir}")
    print(f"Target step:       {args.max_step}")
    print(f"Process chunk:     {args.chunk_steps} steps")
    print(f"Checkpoint prefix: {checkpoint_prefix}")
    if previous:
        print(f"Resuming verified checkpoint: {previous} (step {current})")

    environment = os.environ.copy()
    environment["ACPP_METAL_COMMAND_TIMEOUT_MS"] = str(
        args.command_timeout_seconds * 1000
    )

    while current < args.max_step:
        target = min(current + args.chunk_steps, args.max_step)
        expected_checkpoint = checkpoint_path(
            checkpoint_prefix, target, args.checkpoint_digits
        )
        attempts = args.max_retries + 1

        for attempt in range(1, attempts + 1):
            if expected_checkpoint.exists():
                # This path is dedicated to the supervisor. Without our marker,
                # it is either stale or was left by an interrupted checkpoint.
                if (expected_checkpoint / MARKER_NAME).exists():
                    print(
                        f"[fatal] verified checkpoint collision at "
                        f"{expected_checkpoint}; resume it or use a fresh prefix",
                        file=sys.stderr,
                    )
                    return 2
                try:
                    quarantined = quarantine_checkpoint(
                        expected_checkpoint, state_dir
                    )
                except RuntimeError as error:
                    print(f"[fatal] {error}", file=sys.stderr)
                    return 2
                print(f"[quarantined] incomplete checkpoint: {quarantined}")

            command = [
                str(executable),
                str(input_file),
                *forwarded_arguments,
                f"max_step={target}",
                f"diagnostics.diags_names={' '.join(diagnostic_names)}",
                f"{args.diagnostic_name}.diag_type=Full",
                f"{args.diagnostic_name}.format=checkpoint",
                f"{args.diagnostic_name}.intervals={target}",
                f"{args.diagnostic_name}.dump_last_timestep=1",
                f"{args.diagnostic_name}.file_prefix={checkpoint_prefix}",
                f"{args.diagnostic_name}.file_min_digits={args.checkpoint_digits}",
                "warpx.verbose=1",
            ]
            if previous:
                command.append(f"amr.restart={previous}")

            log_path = state_dir / (
                f"step-{current:010d}-to-{target:010d}-attempt-{attempt}.log"
            )
            print(
                f"[chunk {current}->{target}] attempt {attempt}/{attempts}; "
                f"log: {log_path}"
            )
            result = run_attempt(
                command,
                log_path,
                environment,
                args.stall_timeout_seconds,
                work_dir,
            )

            if (
                result.returncode == 0
                and not result.watchdog_fired
                and not result.metal_timeout
            ):
                try:
                    mark_checkpoint(
                        expected_checkpoint, target, previous, command
                    )
                except RuntimeError as error:
                    print(f"[fatal] {error}", file=sys.stderr)
                    return 2
                previous = expected_checkpoint.resolve()
                current = target
                print(f"[committed] verified checkpoint: {previous}")
                break

            recoverable_signal = result.returncode in RECOVERABLE_RETURN_CODES
            recoverable = (
                result.watchdog_fired
                or result.metal_timeout
                or recoverable_signal
            )
            reason = (
                "no-log-progress watchdog"
                if result.watchdog_fired
                else "Metal completion timeout"
                if result.metal_timeout
                else f"child terminated by signal {-result.returncode}"
                if recoverable_signal
                else f"child exit {result.returncode}"
            )
            print(f"[failed] {reason}", file=sys.stderr)
            if not recoverable:
                print(
                    f"[fatal] non-recoverable WarpX failure; inspect {log_path}",
                    file=sys.stderr,
                )
                return result.returncode or 2
            if attempt == attempts:
                print(
                    f"[fatal] exhausted {args.max_retries} retries for "
                    f"chunk {current}->{target}",
                    file=sys.stderr,
                )
                return 75
            if args.retry_backoff_seconds:
                time.sleep(args.retry_backoff_seconds)
        else:
            return 75

    print(f"[complete] reached step {current}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
