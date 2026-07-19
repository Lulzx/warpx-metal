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


if __name__ == "__main__":
    unittest.main()
