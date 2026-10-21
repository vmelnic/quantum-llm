"""Exercise the Bash launcher with a file-only fake Windows transport."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class ModelLauncherMtpPolicyTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        ops = self.root / "ops"
        ops.mkdir()
        source = Path(__file__).resolve().parents[1] / "ops"
        for name in ("model.sh", "model-aliases.tsv"):
            shutil.copy2(source / name, ops / name)
        (self.root / ".env").write_text(
            "MODEL_MTP_DRAFT_DEPTH=6\nMODEL_MAX_CONTEXT=262144\n"
            "QUANTUM_LLM_REMOTE=fake-host\nQUANTUM_LLM_REMOTE_ROOT=fake-repo\n"
            f"MODEL_ROOT={self.root / 'models'}\n",
            encoding="utf-8",
        )
        self.calls = self.root / "calls.jsonl"
        self.contract_path = self.root / "contract.json"
        # These commands only record arguments; they cannot access the host,
        # invoke PowerShell, inspect processes, or start an inference service.
        transport = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ["FAKE_CALLS"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
if sys.argv[1] == "Get-ModelArtifactContract.ps1":
    print(Path(os.environ["FAKE_CONTRACT"]).read_text())
else:
    print('{"status":"fake"}')
'''
        (ops / "run-on-windows-host.sh").write_text(transport, encoding="utf-8")
        (ops / "run-on-windows-host.sh").chmod(0o755)
        (ops / "sync-to-windows-host.sh").write_text(
            '#!/usr/bin/env bash\nexec "$(dirname "$0")/run-on-windows-host.sh" sync "$@"\n',
            encoding="utf-8",
        )
        (ops / "sync-to-windows-host.sh").chmod(0o755)

    def launch(self, selector, contract, override=None):
        self.contract_path.write_text(json.dumps(contract), encoding="utf-8")
        env = {"PATH": os.environ["PATH"], "HOME": str(self.root),
               "FAKE_CALLS": str(self.calls), "FAKE_CONTRACT": str(self.contract_path)}
        if override is not None:
            env["MODEL_MTP_DRAFT_DEPTH"] = override
        result = subprocess.run(
            ["bash", str(self.root / "ops/model.sh"), "restart", selector],
            env=env, text=True, capture_output=True, check=False,
        )
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        return result, calls

    @staticmethod
    def scalar_contract():
        return {"maximum_context": 1048576, "maximum_thinking_tokens": 0,
                "minimum_exact_kv_bytes_per_token": 23040, "mtp_layers": 0,
                "exact_decode_abi": 0, "exact_decode_draft_depth": 0,
                "exact_decode_draft_vocabulary_size": 0,
                "exact_decode_mtp_kv_encoding": 0,
                "exact_decode_maximum_emitted_tokens": 0}

    @staticmethod
    def mtp_contract():
        return {"maximum_context": 262144, "maximum_thinking_tokens": 32768,
                "minimum_exact_kv_bytes_per_token": 0, "mtp_layers": 1,
                "exact_decode_abi": 2, "exact_decode_draft_depth": 4,
                "exact_decode_draft_vocabulary_size": 65536,
                "exact_decode_mtp_kv_encoding": 1,
                "exact_decode_maximum_emitted_tokens": 5}

    def test_scalar_artifact_ignores_file_default_and_validates_before_stop(self):
        result, calls = self.launch("mistral", self.scalar_contract())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ignoring .env MODEL_MTP_DRAFT_DEPTH=6", result.stdout)
        commands = [call[0] for call in calls]
        self.assertLess(commands.index("Get-ModelArtifactContract.ps1"),
                        commands.index("Stop-ExpertServer.ps1"))
        self.assertEqual(commands.count("Stop-ExpertServer.ps1"), 1)
        install = next(call for call in calls if call[0] == "Install-ExpertServerTask.ps1")
        self.assertNotIn("-WorkerMtpDraftDepth", install)
        self.assertEqual(install[install.index("-MaximumContext") + 1], "262144")
        self.assertEqual(install[install.index("-WorkerKvCacheDtype") + 1], "artifact")

    def test_sampled_mtp_artifact_keeps_six_proposals(self):
        result, calls = self.launch("qwen", self.mtp_contract())
        self.assertEqual(result.returncode, 0, result.stderr)
        install = next(call for call in calls if call[0] == "Install-ExpertServerTask.ps1")
        self.assertEqual(install[install.index("-WorkerMtpDraftDepth") + 1], "6")

    def test_explicit_unsupported_override_fails_without_stopping(self):
        result, calls = self.launch("mistral", self.scalar_contract(), override="6")
        self.assertEqual(result.returncode, 2)
        self.assertIn("invocation override requires", result.stderr)
        self.assertFalse(any(call[0] in ("Stop-ExpertServer.ps1", "Install-ExpertServerTask.ps1")
                             for call in calls))

    def test_depth_is_checked_after_artifact_context_clamp(self):
        contract = self.mtp_contract()
        contract["maximum_context"] = 6
        contract["maximum_thinking_tokens"] = 0
        result, calls = self.launch("qwen", contract)
        self.assertEqual(result.returncode, 2)
        self.assertIn("effective artifact context", result.stderr)
        self.assertFalse(any(call[0] == "Stop-ExpertServer.ps1" for call in calls))


if __name__ == "__main__":
    unittest.main()
