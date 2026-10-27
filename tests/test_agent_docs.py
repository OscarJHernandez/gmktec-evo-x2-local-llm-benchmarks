"""Keep AGENTS.md, the skills and the learnings file consistent with the repository."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SKILLS = sorted((ROOT / ".claude/skills").glob("*/SKILL.md"))
PATH_REFERENCE = re.compile(r"`((?:src|tests|docs|notebooks|\.github|\.claude)/[\w./\-]+)`")


def test_skills_exist() -> None:
    assert {path.parent.name for path in SKILLS} >= {"capture-learnings", "prepare-pr"}


@pytest.mark.parametrize("skill", SKILLS, ids=lambda path: path.parent.name)
def test_skill_frontmatter_matches_directory(skill: Path) -> None:
    match = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", skill.read_text())
    assert match, "SKILL.md needs name and description frontmatter"
    assert match.group(1) == skill.parent.name
    assert len(match.group(2)) > 40


def test_agents_md_lists_every_skill() -> None:
    agents = (ROOT / "AGENTS.md").read_text()
    for skill in SKILLS:
        assert f".claude/skills/{skill.parent.name}/SKILL.md" in agents


@pytest.mark.parametrize(
    "document",
    [ROOT / "AGENTS.md", ROOT / "docs/agent-learnings.md", *SKILLS],
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_referenced_repository_paths_exist(document: Path) -> None:
    for reference in PATH_REFERENCE.findall(document.read_text()):
        path = reference.split("::")[0]
        assert (ROOT / path).exists(), f"{document.name} references missing {path}"


def test_claude_md_imports_agents_md() -> None:
    assert (ROOT / "CLAUDE.md").read_text().strip() == "@AGENTS.md"
