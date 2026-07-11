# Local CI

Run the repository's portability build from the repository root:

```bash
nice -n 15 ./ci/run-local-ci.sh
```

The script clones pinned AMReX and WarpX revisions into a temporary directory,
verifies each pin is reachable from its upstream `26.06` tag, installs the
packaged AMReX replacement files, explicitly applies the AMReX and WarpX patch
files used by this repository, and builds linked 2D and 3D CPU WarpX
executables. Any clone, pin, patch-application, configuration, compilation, or
link failure exits nonzero. The last output line is machine-readable:

```text
LOCAL_CI_RESULT outcome=PASS|FAIL host=... patches=... binaries=...
```

Dependencies are Git, CMake, Ninja, Python 3, a C/C++ compiler, and OpenMP. On
macOS, Homebrew `libomp` is used when available.

This is host-platform CI. On macOS the full build exercises the Apple CPU path.
The script also recompiles the WarpX translation units containing the relevant
`__APPLE__` guards with `-U__APPLE__` and `-fsyntax-only`. That is a useful
guard-leakage check, but it does not reproduce Linux's kernel, libc, ABI,
compiler, linker, packaging, or runtime. Only a true Linux runner proves those
properties. On Linux, the full build exercises the native Linux CPU path.

Set `CMAKE_BUILD_PARALLEL_LEVEL` to change the default two build jobs. Set
`KEEP_LOCAL_CI_WORKDIR=1` to retain the temporary source and build trees for
debugging.
