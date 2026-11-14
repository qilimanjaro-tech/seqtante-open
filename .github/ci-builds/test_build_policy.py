# Copyright 2026 Qilimanjaro Quantum Tech
# ruff: file-ignore[implicit-namespace-package] -- Standalone workflow regression module.
"""Exercise uv's actual build policy through a local artifact index."""

import functools
import hashlib
import http.server
import io
import json
import os
import re
import shlex
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- Test the real uv command at the process interface.
import sys
import tarfile
import tempfile
import threading
import tomllib
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def wheel(directory, name, requirements=()):
    target = directory / name
    target.mkdir(parents=True, exist_ok=True)
    filename = f"{name.replace('-', '_')}-1.0-py3-none-any.whl"
    artifact = target / filename
    info = f"{name.replace('-', '_')}-1.0.dist-info"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(
            f"{info}/METADATA",
            f"Metadata-Version: 2.2\nName: {name}\nVersion: 1.0\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requirements),
        )
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    (target / "index.html").write_text(f'<a href="{filename}#sha256={digest.split(":", 1)[1]}">{filename}</a>')
    return digest


class BuildHashTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.project = self.work / "project"
        self.project.mkdir()
        index = self.work / "index"
        index.mkdir()
        self.artifacts = index
        self.hashes = {
            "transitive-helper": wheel(index, "transitive-helper"),
            "build-helper": wheel(index, "build-helper", ["transitive-helper==1.0"]),
        }

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(index))
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.index = f"http://127.0.0.1:{server.server_port}"
        self.marker = self.work / "backend-executed"
        (self.project / "backend.py").write_text(
            "from pathlib import Path\nimport zipfile\n"
            f"Path({str(self.marker)!r}).touch()\n"
            "def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):\n"
            "    filename = 'fixture_project-1.0-py3-none-any.whl'\n"
            "    with zipfile.ZipFile(Path(wheel_directory) / filename, 'w') as archive:\n"
            "        archive.writestr('fixture_project-1.0.dist-info/METADATA', 'Metadata-Version: 2.2\\nName: fixture-project\\nVersion: 1.0\\n')\n"
            "        archive.writestr('fixture_project-1.0.dist-info/WHEEL', 'Wheel-Version: 1.0\\nRoot-Is-Purelib: true\\nTag: py3-none-any\\n')\n"
            "        archive.writestr('fixture_project-1.0.dist-info/RECORD', '')\n"
            "    return filename\n"
        )

    def build(self, constrained):
        policy = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["uv"]
        constraints = [{"requirement": f"{name}==1.0", "hashes": [self.hashes[name]]} for name in constrained]
        # Use the repository's real enforcement switch, with an independent
        # fixture dependency graph so missing transitive hashes are observable.
        configuration = (
            '[project]\nname = "fixture-project"\nversion = "1.0"\n'
            '[build-system]\nrequires = ["build-helper==1.0"]\nbuild-backend = "backend"\nbackend-path = ["."]\n'
            '[tool.uv]\npreview-features = ["build-dependency-hashes"]\n'
            f"require-build-hashes = {str(policy.get('require-build-hashes', False)).lower()}\n"
            "build-constraint-dependencies = [\n"
            + "\n".join(
                f'{{ requirement = "{item["requirement"]}", hashes = {json.dumps(item["hashes"])} }},'
                for item in constraints
            )
            + f'\n]\n[[tool.uv.index]]\nurl = "{self.index}"\ndefault = true\n'
        )
        (self.project / "pyproject.toml").write_text(configuration)
        return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- Selected uv, trusted first-party fixture and loopback index.
            [shutil.which("uv"), "build", "--wheel", "--no-build", "--no-cache", "--python", sys.executable],
            cwd=self.project,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_unhashed_transitive_build_dependency_fails_before_backend_execution(self):
        result = self.build(["build-helper"])
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.marker.exists())
        self.assertIn("hash", result.stderr.lower())

    def test_complete_build_dependency_hashes_allow_first_party_build(self):
        result = self.build(["build-helper", "transitive-helper"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.marker.exists())
        self.assertTrue((self.project / "dist/fixture_project-1.0-py3-none-any.whl").exists())

    def test_wrong_build_dependency_hash_fails_before_backend_execution(self):
        self.hashes["build-helper"] = "sha256:" + "0" * 64
        result = self.build(["build-helper", "transitive-helper"])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists())
        self.assertIn("hash", result.stderr.lower())

    def test_ci_install_denies_an_unlisted_source_backend(self):
        wheel(self.artifacts, "unapproved-source")
        (self.project / "pyproject.toml").write_text(
            '[project]\nname = "fixture-project"\nversion = "1.0"\nrequires-python = ">=3.11,<3.14"\n'
            'dependencies = ["unapproved-source==1.0"]\n[dependency-groups]\ndev = []\n'
            f'[tool.uv]\npackage = false\n[[tool.uv.index]]\nurl = "{self.index}"\ndefault = true\n'
        )
        env = {**os.environ, "UV_PYTHON": sys.executable, "UV_PROJECT_ENVIRONMENT": str(self.project / ".venv")}
        locked = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- Generate the fixture lock without source execution.
            [shutil.which("uv"), "lock", "--no-build", "--no-cache"],
            cwd=self.project,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(locked.returncode, 0, locked.stderr)
        archive_path = self.artifacts / "unapproved-source/unapproved_source-1.0.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            files = {
                "PKG-INFO": "Metadata-Version: 2.2\nName: unapproved-source\nVersion: 1.0\nRequires-Python: >=3.11\n",
                "pyproject.toml": '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n',
                "backend.py": f"from pathlib import Path\nPath({str(self.marker)!r}).touch()\n",
            }
            for name, content in files.items():
                member = tarfile.TarInfo(f"unapproved_source-1.0/{name}")
                data = content.encode()
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        # Keep fixed runtime metadata and substitute the independent source fixture.
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        lock = self.project / "uv.lock"
        lock.write_text(
            re.sub(
                r"wheels = \[.*?\]\n",
                f'sdist = {{ url = "{self.index}/unapproved-source/{archive_path.name}", hash = "sha256:{digest}" }}\n',
                lock.read_text(),
                count=1,
                flags=re.DOTALL,
            )
        )
        command = next(
            line.partition("run: ")[2]
            for line in (ROOT / ".github/workflows/code_quality.yml").read_text().splitlines()
            if line.strip().startswith("run: uv sync ")
        )
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- Run the actual single-line CI install against a loopback fixture.
            shlex.split(command), cwd=self.project, env=env, capture_output=True, text=True, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists())
        self.assertIn("build", result.stderr.lower())


class ExceptionPolicyTests(unittest.TestCase):
    def check(self, contents):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "uv.lock"
            lock.write_text(contents)
            return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- Exercise the policy's public command with a temporary lock.
                [sys.executable, str(ROOT / ".github/ci-builds/check_policy.py"), str(lock)],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_approved_source_exception_is_accepted(self):
        result = self.check((ROOT / "uv.lock").read_text())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_source_hash_is_rejected(self):
        lock = (
            (ROOT / "uv.lock")
            .read_text()
            .replace("798528d69253c64f6e70b283c623a7cb6b5265806320bd90cec424a9f384f5ff", "0" * 64)
        )
        result = self.check(lock)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Qblox", result.stderr)

    def test_changed_exception_version_is_rejected(self):
        lock = (
            (ROOT / "uv.lock")
            .read_text()
            .replace('name = "qblox-instruments"\nversion = "1.0.3"', 'name = "qblox-instruments"\nversion = "1.0.4"')
        )
        result = self.check(lock)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Qblox", result.stderr)

    def test_unreviewed_local_dependency_is_rejected(self):
        lock = (ROOT / "uv.lock").read_text() + '\n[[package]]\nname = "unexpected"\nsource = { path = "../other" }\n'
        result = self.check(lock)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Unreviewed non-registry source", result.stderr)


if __name__ == "__main__":
    unittest.main()
