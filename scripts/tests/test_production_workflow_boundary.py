from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTALLED = ROOT / ".github/workflows"


class ProductionWorkflowBoundaryTests(unittest.TestCase):
    def test_production_workflows_do_not_reference_the_retired_research_tree(self) -> None:
        for path in sorted(INSTALLED.glob("*.yml")):
            with self.subTest(workflow=path.name):
                self.assertNotIn("autoresearch", path.read_text(encoding="utf-8").lower())

    def test_automatic_ci_only_runs_at_pr_readiness(self) -> None:
        for path in sorted(INSTALLED.iterdir()):
            if path.suffix not in {".yaml", ".yml"} or path.name == "package-release.yml":
                continue
            with self.subTest(workflow=path.name):
                source = path.read_text(encoding="utf-8")
                triggers = source.split("\non:\n", 1)[1].split("\n\n", 1)[0]
                events = set(re.findall(r"^  ([\w_]+):", triggers, re.M))
                self.assertTrue(events and events <= {"pull_request", "workflow_dispatch"}, events)
                if "pull_request" not in events:
                    continue
                self.assertIn("  pull_request:\n    types: [opened, reopened, ready_for_review]", triggers)
                jobs = re.split(r"^  [\w-]+:\s*$", source.split("\njobs:\n", 1)[1], flags=re.M)[1:]
                self.assertTrue(jobs)
                for job in jobs:
                    header = job.split("    steps:", 1)[0]
                    if "    needs:" in header and "always()" not in header:
                        continue  # A skipped prerequisite skips its dependants.
                    condition = re.search(r"^    if: (.*(?:\n      [^\n]+)*)", header, re.M)
                    self.assertIsNotNone(condition)
                    expression = condition.group(1)
                    self.assertTrue(
                        "github.event.pull_request.draft == false" in expression
                        or ("github.event_name == 'workflow_dispatch'" in expression
                            and "pull_request" not in expression),
                        expression,
                    )

    def test_package_release_is_manual_or_tag_only(self) -> None:
        source = (INSTALLED / "package-release.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", source)
        self.assertIn('tags:\n      - "v*"', source)
        self.assertNotIn("pull_request:", source)
        self.assertNotIn("schedule:", source)


if __name__ == "__main__":
    unittest.main()
