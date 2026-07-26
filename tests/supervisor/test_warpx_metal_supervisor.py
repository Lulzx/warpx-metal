#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_PATH = ROOT / "scripts" / "10-run-warpx-resilient.py"


def load_supervisor():
    spec = importlib.util.spec_from_file_location(
        "warpx_metal_supervisor", SUPERVISOR_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUPERVISOR = load_supervisor()


class WarpXMetalSupervisorTest(unittest.TestCase):
    def test_reads_diagnostics_through_input_include(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.inputs").write_text(
                "diagnostics.diags_names = diag1 particles\n",
                encoding="utf-8",
            )
            (root / "main.inputs").write_text(
                "FILE = base.inputs\n",
                encoding="utf-8",
            )
            self.assertEqual(
                SUPERVISOR.read_diagnostic_names(root / "main.inputs"),
                ["diag1", "particles"],
            )

    def test_retries_timeout_and_commits_only_successful_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_file = root / "inputs"
            input_file.write_text(
                "diagnostics.diags_names = diag1\n", encoding="utf-8"
            )
            fake_warpx = root / "fake-warpx.py"
            fake_warpx.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import os
                    import pathlib
                    import signal
                    import sys

                    assignments = {
                        item.split("=", 1)[0]: item.split("=", 1)[1]
                        for item in sys.argv[1:]
                        if "=" in item
                    }
                    target = int(assignments["max_step"])
                    prefix = pathlib.Path(
                        assignments["metal_recovery_chk.file_prefix"]
                    )
                    digits = int(
                        assignments["metal_recovery_chk.file_min_digits"]
                    )
                    checkpoint = prefix.parent / f"{prefix.name}{target:0{digits}d}"
                    attempt = pathlib.Path.cwd() / f".attempt-{target}"
                    if target == 2 and not attempt.exists():
                        attempt.write_text("failed once")
                        checkpoint.mkdir(parents=True)
                        (checkpoint / "WarpXHeader").write_text("partial")
                        print(
                            "metal_queue: kernel timed out after 10 ms; "
                            "command buffer remained scheduled"
                        )
                        raise SystemExit(70)
                    if target == 4 and not attempt.exists():
                        attempt.write_text("terminated once")
                        os.kill(os.getpid(), signal.SIGTERM)

                    checkpoint.mkdir(parents=True)
                    (checkpoint / "WarpXHeader").write_text("header")
                    (checkpoint / "Level_0").mkdir()
                    print(f"STEP {target}")
                    """
                ),
                encoding="utf-8",
            )
            fake_warpx.chmod(0o755)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR_PATH),
                    "--max-step",
                    "4",
                    "--chunk-steps",
                    "2",
                    "--max-retries",
                    "1",
                    "--retry-backoff-seconds",
                    "0",
                    "--stall-timeout-seconds",
                    "10",
                    "--work-dir",
                    str(root),
                    "--checkpoint-prefix",
                    "checkpoints/chk",
                    str(fake_warpx),
                    str(input_file),
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            checkpoint_two = root / "checkpoints" / "chk0000000002"
            checkpoint_four = root / "checkpoints" / "chk0000000004"
            marker_two = json.loads(
                (checkpoint_two / SUPERVISOR.MARKER_NAME).read_text(
                    encoding="utf-8"
                )
            )
            marker_four = json.loads(
                (checkpoint_four / SUPERVISOR.MARKER_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(marker_two["step"], 2)
            self.assertEqual(marker_four["step"], 4)
            self.assertEqual(
                Path(marker_four["previous_checkpoint"]), checkpoint_two.resolve()
            )
            self.assertEqual(
                len(list((root / ".warpx-metal-supervisor").glob("*.log"))),
                4,
            )
            self.assertEqual(
                len(
                    list(
                        (
                            root
                            / ".warpx-metal-supervisor"
                            / "quarantine"
                        ).iterdir()
                    )
                ),
                1,
            )

    def test_does_not_retry_non_metal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_file = root / "inputs"
            input_file.write_text("", encoding="utf-8")
            fake_warpx = root / "fake-failure.py"
            fake_warpx.write_text(
                "#!/usr/bin/env python3\n"
                "print('invalid input deck')\n"
                "raise SystemExit(9)\n",
                encoding="utf-8",
            )
            fake_warpx.chmod(0o755)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SUPERVISOR_PATH),
                    "--max-step",
                    "2",
                    "--chunk-steps",
                    "2",
                    "--max-retries",
                    "3",
                    "--retry-backoff-seconds",
                    "0",
                    "--work-dir",
                    str(root),
                    str(fake_warpx),
                    str(input_file),
                ],
                text=True,
                capture_output=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 9)
            self.assertEqual(
                len(list((root / ".warpx-metal-supervisor").glob("*.log"))),
                1,
            )


class CpuDemotionTest(unittest.TestCase):
    """The CPU build has no Metal dependency, so it is the absorbing state the
    run falls back to when the GPU keeps wedging."""

    WEDGING_GPU = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        print(
            "metal_queue: producer before device-to-host memcpy timed out "
            "after 120000 ms; command buffer remained committed"
        )
        raise SystemExit(70)
        """
    )

    WORKING_CPU = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import pathlib
        import sys

        assignments = {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in sys.argv[1:]
            if "=" in item
        }
        target = int(assignments["max_step"])
        prefix = pathlib.Path(assignments["metal_recovery_chk.file_prefix"])
        digits = int(assignments["metal_recovery_chk.file_min_digits"])
        checkpoint = prefix.parent / f"{prefix.name}{target:0{digits}d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "WarpXHeader").write_text("header")
        (checkpoint / "Level_0").mkdir()
        print(f"STEP {target}")
        """
    )

    def _write(self, path: Path, body: str) -> Path:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        return path

    def _run(self, root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        input_file = root / "inputs"
        input_file.write_text(
            "diagnostics.diags_names = diag1\n", encoding="utf-8"
        )
        return subprocess.run(
            [
                sys.executable,
                str(SUPERVISOR_PATH),
                "--max-step", "2",
                "--chunk-steps", "2",
                "--max-retries", "1",
                "--retry-backoff-seconds", "0",
                "--stall-timeout-seconds", "10",
                "--work-dir", str(root),
                "--checkpoint-prefix", "checkpoints/chk",
                *extra,
                str(root / "gpu.py"),
                str(input_file),
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )

    def test_demotes_to_cpu_after_persistent_metal_wedge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root / "gpu.py", self.WEDGING_GPU)
            cpu = self._write(root / "cpu.py", self.WORKING_CPU)

            result = self._run(
                root,
                "--cpu-fallback-executable", str(cpu),
                "--cpu-fallback-retries", "0",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("[demoted]", result.stderr)

            checkpoint = root / "checkpoints" / "chk0000000002"
            marker = json.loads(
                (checkpoint / SUPERVISOR.MARKER_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(marker["step"], 2)
            # The committed checkpoint must have been produced by the CPU
            # binary, not merely attributed to it.
            self.assertEqual(Path(marker["command"][0]).resolve(), cpu.resolve())

            logs = sorted(
                p.name
                for p in (root / ".warpx-metal-supervisor").glob("*.log")
            )
            # Two wedged GPU attempts, then one CPU attempt.
            self.assertEqual(len(logs), 3, logs)
            self.assertEqual(sum("-gpu-" in name for name in logs), 2, logs)
            self.assertEqual(sum("-cpu-" in name for name in logs), 1, logs)

    def test_does_not_demote_when_failure_is_not_a_gpu_wedge(self) -> None:
        """An external SIGTERM is retryable but says nothing about the GPU.
        Demoting on it would hide real failures behind a silent slowdown."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(
                root / "gpu.py",
                "#!/usr/bin/env python3\n"
                "import os, signal\n"
                "os.kill(os.getpid(), signal.SIGTERM)\n",
            )
            cpu = self._write(root / "cpu.py", self.WORKING_CPU)

            result = self._run(
                root, "--cpu-fallback-executable", str(cpu)
            )
            self.assertEqual(result.returncode, 75)
            self.assertNotIn("[demoted]", result.stderr)

    def test_rejects_unusable_cpu_fallback_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write(root / "gpu.py", self.WORKING_CPU)

            result = self._run(
                root,
                "--cpu-fallback-executable", str(root / "missing.py"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cpu fallback is missing", result.stderr)
            # Must fail before any child runs, not at demotion time.
            self.assertFalse((root / ".warpx-metal-supervisor").exists())


if __name__ == "__main__":
    unittest.main()
