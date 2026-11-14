# Copyright 2026 Qilimanjaro Quantum Tech
# ruff: file-ignore[implicit-namespace-package] -- Standalone unittest module, not an importable package.
"""Exercise the workflow commands with local indexes and publication adapters."""

import functools
import hashlib
import http.server
import io
import json
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- Execute the workflow scripts and local test tools.
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())
STEPS = WORKFLOW["jobs"]["deploy-codeartifacts"]["steps"]


def script(name):
    return next(step["run"] for step in STEPS if step.get("name") == name)


def run_shell(command, work, env):
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- Scripts come from the checked-out workflow.
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", command],  # ruff: ignore[start-process-with-partial-path] -- CI provides bash.
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def write_wheel(directory, version="1.0"):
    wheel = directory / f"fixture_dep-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"fixture_dep-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.2\nName: fixture-dep\nVersion: {version}\n"
            "Summary: Offline test artifact\nDescription-Content-Type: text/plain\n\nTest package.\n",
        )
        archive.writestr(
            f"fixture_dep-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"fixture_dep-{version}.dist-info/RECORD", "")
    return wheel


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        (self.work / "dist").mkdir()
        for filename in ("example-1.0-py3-none-any.whl", "example-1.0.tar.gz"):
            (self.work / "dist" / filename).touch()
        self.interpreter = self.work / "tool-python"
        self.interpreter.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv) + '\\n')\n"
            "if sys.argv[3] == 'check':\n"
            "    sys.exit(int(os.environ.get('CHECK_EXIT', '0')))\n"
        )
        self.interpreter.chmod(0o755)
        commands = self.work / "bin"
        commands.mkdir()
        aws = commands / "aws"
        aws.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$AWS_CALLS"\nprintf "%s\\n" fixture-value\n')
        aws.chmod(0o755)
        uvx = commands / "uvx"
        uvx.write_text("#!/bin/sh\necho 'Transient tool resolution is prohibited' >&2\nexit 99\n")
        uvx.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "RELEASE_TOOLS_PYTHON": str(self.interpreter),
            "CALLS": str(self.work / "calls.jsonl"),
            "AWS_CALLS": str(self.work / "aws-calls"),
            "CA_DOMAIN": "fixture",
            "CA_DOMAIN_OWNER": "fixture",
            "CA_REPOSITORY": "fixture",
        }

    def publish(self, dry_run=False):
        validation = next(step for step in STEPS if step.get("name") == "List & validate distributions")
        self.assertFalse(validation.get("continue-on-error", False))
        result = run_shell(validation["run"], self.work, self.env)
        upload = next(step for step in STEPS if step.get("name") == "Publish to CodeArtifact")
        # GitHub supplies the default success() gate for this conditional step.
        self.assertEqual(upload["if"], "${{ !inputs.dry_run }}")
        if result.returncode == 0 and not dry_run:
            result = run_shell(upload["run"], self.work, self.env)
        return result

    def calls(self):
        return [json.loads(line) for line in (self.work / "calls.jsonl").read_text().splitlines()]

    def test_validation_and_upload_use_one_environment(self):
        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        artifacts = ["dist/example-1.0-py3-none-any.whl", "dist/example-1.0.tar.gz"]
        self.assertEqual(
            self.calls(),
            [
                [str(self.interpreter), "-m", "twine", "check", *artifacts],
                [str(self.interpreter), "-m", "twine", "upload", *artifacts],
            ],
        )

    def test_dry_run_checks_without_upload_token_or_endpoint(self):
        result = self.publish(dry_run=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse((self.work / "aws-calls").exists())

    def test_failed_validation_blocks_upload(self):
        self.env["CHECK_EXIT"] = "1"
        result = self.publish()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse((self.work / "aws-calls").exists())

    def test_real_twine_checks_valid_wheel_and_sdist_and_rejects_bad_metadata(self):
        for artifact in (self.work / "dist").iterdir():
            artifact.unlink()
        write_wheel(self.work / "dist")
        with tarfile.open(self.work / "dist/fixture_dep-1.0.tar.gz", "w:gz") as archive:
            root = tarfile.TarInfo("fixture_dep-1.0")
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            metadata = b"Metadata-Version: 2.2\nName: fixture-dep\nVersion: 1.0\nSummary: Fixture\n\nTest package.\n"
            member = tarfile.TarInfo("fixture_dep-1.0/PKG-INFO")
            member.size = len(metadata)
            archive.addfile(member, io.BytesIO(metadata))
        self.env["RELEASE_TOOLS_PYTHON"] = sys.executable
        valid = run_shell(script("List & validate distributions"), self.work, self.env)
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        with zipfile.ZipFile(self.work / "dist/bad-1.0-py3-none-any.whl", "w") as archive:
            archive.writestr("bad-1.0.dist-info/METADATA", "Metadata-Version: 2.2\nSummary: Missing name and version\n")
        invalid = run_shell(script("List & validate distributions"), self.work, self.env)
        self.assertNotEqual(invalid.returncode, 0)

    def test_tool_source_is_the_workflow_revision_even_for_an_old_build_ref(self):
        build_checkout = WORKFLOW["jobs"]["build"]["steps"][0]["with"]
        tool_checkout = next(step["with"] for step in STEPS if step.get("name") == "Check out release tools")
        self.assertEqual(build_checkout["ref"], "${{ inputs.ref || github.ref }}")
        self.assertEqual(tool_checkout["ref"], "${{ github.workflow_sha }}")
        self.assertFalse(tool_checkout["persist-credentials"])
        self.assertEqual(build_checkout["fetch-depth"], 0)


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.project = self.work / "release-tools-source/.github/release-tools"
        self.project.mkdir(parents=True)
        index = self.work / "index"
        self.wheels = index / "fixture-dep"
        self.wheels.mkdir(parents=True)

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
        self.index_url = f"http://127.0.0.1:{server.server_port}"
        self.manifest = self.project / "pyproject.toml"
        self.manifest.write_text(
            '[project]\nname = "fixture-tools"\nversion = "0"\nrequires-python = ">=3.13,<3.14"\n'
            'dependencies = ["fixture-dep>=1.0"]\n[tool.uv]\npackage = false\n'
            f'[[tool.uv.index]]\nurl = "{self.index_url}"\ndefault = true\n'
        )
        self.env = {
            **os.environ,
            "UV_CACHE_DIR": str(self.work / "cache"),
            "UV_PYTHON": sys.executable,
            "UV_PROJECT_ENVIRONMENT": str(self.project / ".venv"),
            "UV_PYTHON_DOWNLOADS": "never",
        }
        self.uv = shutil.which("uv")
        self.assertIsNotNone(self.uv)

    def lock(self):
        self.refresh_index()
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- Invoke the selected uv against a loopback fixture index.
            [self.uv, "lock", "--project", str(self.project), "--no-build"],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def refresh_index(self):
        links = []
        for artifact in self.wheels.iterdir():
            if artifact.suffix in {".whl", ".gz"}:
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                links.append(f'<a href="{artifact.name}#sha256={digest}">{artifact.name}</a>')
        (self.wheels / "index.html").write_text("\n".join(links))

    def install(self):
        # Each attempt starts without an installed environment or a warm artifact cache.
        shutil.rmtree(self.project / ".venv", ignore_errors=True)
        shutil.rmtree(self.work / "cache", ignore_errors=True)
        return run_shell(script("Install release tools"), self.work, self.env)

    def test_newer_index_candidate_does_not_change_the_locked_environment(self):
        write_wheel(self.wheels)
        self.lock()
        lock = (self.project / "uv.lock").read_bytes()
        write_wheel(self.wheels, "2.0")
        self.refresh_index()
        for _ in range(2):
            result = self.install()
            self.assertEqual(result.returncode, 0, result.stderr)
            version = subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true] -- Read metadata from the temporary fixture environment.
                [
                    str(self.project / ".venv/bin/python"),
                    "-c",
                    "from importlib.metadata import version; print(version('fixture-dep'))",
                ],
                text=True,
            ).strip()
            self.assertEqual(version, "1.0")
        self.assertEqual((self.project / "uv.lock").read_bytes(), lock)

    def test_stale_or_missing_lock_fails_without_updating_it(self):
        write_wheel(self.wheels)
        self.lock()
        lock = (self.project / "uv.lock").read_bytes()
        self.manifest.write_text(self.manifest.read_text().replace("fixture-dep>=1.0", "fixture-dep==2.0"))
        self.assertNotEqual(self.install().returncode, 0)
        self.assertEqual((self.project / "uv.lock").read_bytes(), lock)
        (self.project / "uv.lock").unlink()
        self.assertNotEqual(self.install().returncode, 0)
        self.assertFalse((self.project / "uv.lock").exists())

    def test_corrupted_locked_wheel_is_rejected(self):
        wheel = write_wheel(self.wheels)
        self.lock()
        original = hashlib.sha256(wheel.read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("tampered.txt", "Changed after lock review")
        self.assertNotEqual(hashlib.sha256(wheel.read_bytes()).hexdigest(), original)
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hash", result.stderr.lower())

    def test_source_only_dependency_is_rejected_before_its_backend_runs(self):
        write_wheel(self.wheels)
        self.lock()
        (self.wheels / "fixture_dep-1.0-py3-none-any.whl").unlink()
        marker = self.work / "backend-executed"
        source = self.wheels / "fixture_dep-1.0.tar.gz"
        with tarfile.open(source, "w:gz") as archive:
            files = {
                "PKG-INFO": "Metadata-Version: 2.2\nName: fixture-dep\nVersion: 1.0\nRequires-Python: >=3.13\n",
                "pyproject.toml": '[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n',
                "backend.py": f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
            }
            for name, content in files.items():
                member = tarfile.TarInfo(f"fixture_dep-1.0/{name}")
                data = content.encode()
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        # Replace the fixture's wheel entry with an exact source artifact, without
        # asking the resolver to execute a backend to generate test metadata.
        lockfile = self.project / "uv.lock"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        lockfile.write_text(
            re.sub(
                r"wheels = \[.*?\]\n",
                f'sdist = {{ url = "{self.index_url}/fixture-dep/{source.name}", hash = "sha256:{source_hash}" }}\n',
                lockfile.read_text(),
                count=1,
                flags=re.DOTALL,
            )
        )
        self.refresh_index()
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(marker.exists())
        self.assertIn("build", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
