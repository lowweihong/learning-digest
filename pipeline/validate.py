#!/usr/bin/env python3
"""Structural checks for content/ — run before every commit.

- every lesson has the required frontmatter and sections
- every lesson has at least 4 review questions, each with a folded answer
- every concept listed in a lesson's frontmatter exists as content/concepts/<name>.md
- every [[wikilink]] in content/ resolves to a file
- every concept note has the required sections and links back to at least one lesson
Exit code 1 on any error.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = Path(os.environ.get("LD_CONTENT", ROOT / "content"))
LESSON_KEYS = {"title", "source", "channel", "date", "concepts", "tags"}
LESSON_SECTIONS = ["## TL;DR", "## Concept map", "## First-principles rebuild", "## Connections", "## Review questions", "## Source notes"]
CONCEPT_KEYS = {"title", "aliases", "tags"}
CONCEPT_SECTIONS = ["## Definition", "## Why it's true", "## Minimal example", "## Prerequisites", "## Appears in"]
WIKILINK = re.compile(r"(?<!!)\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")


def split_fm(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    return (yaml.safe_load(text[4:end]) or {}), text[end + 4:]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def resolve(link: str, files: dict[str, Path]) -> bool:
    link = link.strip().removesuffix(".md")
    return link in files or link.split("/")[-1] in {k.split("/")[-1] for k in files}


def main() -> int:
    errors: list[str] = []
    files = {str(p.relative_to(CONTENT).with_suffix("")): p for p in CONTENT.rglob("*.md")}

    for rel, path in files.items():
        fm, body = split_fm(path.read_text())
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        where = f"content/{rel}.md"

        for link in WIKILINK.findall(body):
            if not resolve(link, files):
                errors.append(f"{where}: broken wikilink [[{link}]]")

        if rel.startswith("lessons/") and rel != "lessons/index":
            if missing := LESSON_KEYS - fm.keys():
                errors.append(f"{where}: frontmatter missing {sorted(missing)}")
            for s in LESSON_SECTIONS:
                if s not in body:
                    errors.append(f"{where}: missing section '{s}'")
            rq = body.split("## Review questions", 1)[-1].split("\n## ", 1)[0]
            n_q = len(re.findall(r"^> \[!question\]-", rq, re.M))
            if n_q < 4:
                errors.append(f"{where}: {n_q} review questions (need >= 4, as '> [!question]-' callouts)")
            for c in fm.get("concepts") or []:
                if f"concepts/{slugify(c)}" not in files:
                    errors.append(f"{where}: concept '{c}' has no note at content/concepts/{slugify(c)}.md")

        elif rel.startswith("concepts/") and rel != "concepts/index":
            if missing := CONCEPT_KEYS - fm.keys():
                errors.append(f"{where}: frontmatter missing {sorted(missing)}")
            for s in CONCEPT_SECTIONS:
                if s not in body:
                    errors.append(f"{where}: missing section '{s}'")
            appears = body.split("## Appears in", 1)[-1]
            if not re.search(r"\[\[lessons/", appears):
                errors.append(f"{where}: 'Appears in' links no lesson")

    for e in errors:
        print("ERROR", e)
    n_lessons = sum(r.startswith("lessons/") and r != "lessons/index" for r in files)
    n_concepts = sum(r.startswith("concepts/") and r != "concepts/index" for r in files)
    print(f"{'FAIL' if errors else 'OK'}: {n_lessons} lessons, {n_concepts} concepts, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
