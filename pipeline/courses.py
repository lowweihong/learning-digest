#!/usr/bin/env python3
"""Build one hub page per course (content/courses/<slug>.md) from the course's own website.

For every `course_page` source in sources.yaml this scrapes the schedule (lecture number,
date, title, description, slides/video/readings links) and writes a Markdown hub that
links each lecture to its lesson once the daily run has written one.

Everything on a hub page comes from the course website or state/processed.json, so
re-running is safe and idempotent. Hubs are generated: don't edit them by hand.

Usage: python3 pipeline/courses.py [--json]   (--json also dumps work/courses/<slug>.json)
"""
from __future__ import annotations

import argparse
import html as H
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from digest import ROOT, SOURCES, WORK, direct_pdf, load_state, video_key  # noqa: E402

OUT = ROOT / "content" / "courses"
NOT_A_LECTURE = re.compile(r"holiday|thanksgiving|no class|recess|break$", re.I)
MONTH_DATE = re.compile(r"^(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\.?\s+)?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2}$")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 learning-digest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def text(fragment: str) -> str:
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()



def classify(link: str, lec: dict) -> None:
    link = H.unescape(link)
    if re.search(r"youtube\.com|youtu\.be", link):
        lec.setdefault("video", f"https://www.youtube.com/watch?v={video_key(link)}" if len(video_key(link)) == 11 else link)
    elif re.search(r"\.pdf(\?|$)", link, re.I) and re.search(r"slide|lec", link, re.I):
        lec.setdefault("slides", link)  # viewer link for readers; digest.py converts to direct download
    elif "colab.research.google.com" in link:
        lec.setdefault("lab", link)


# ------------------------------------------------------------------ parsers

def parse_table(html: str) -> list[dict]:
    """Schedule as an HTML table: Week | Date | Lecture (title, description, slides) | Material."""
    lectures, section = [], ""
    for row in re.findall(r"<tr.*?</tr>", html, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 3 or not MONTH_DATE.match(text(cells[1])):
            continue
        cell = cells[2]
        title = re.search(r'class="lec-title"[^>]*>(.*?)</', cell, re.S)
        if not title:
            continue
        desc = re.search(r'class="lec-desc"[^>]*>(.*?)</span>', cell, re.S)
        theme = re.search(r'class="theme-tag"[^>]*>(.*?)</', cell, re.S)
        if theme:  # the theme tag only marks the first lecture of each section
            section = text(theme.group(1))
        lec = {"date": text(cells[1]), "title": text(title.group(1)),
               "description": text(desc.group(1)) if desc else "", "section": section}
        for link in re.findall(r'href="([^"]+)"', cell):
            classify(link, lec)
        if len(cells) > 3 and text(cells[3]):
            items = [f"[{text(t)}]({H.unescape(h)})" for h, t in re.findall(r'<a [^>]*href="([^"]+)"[^>]*>(.*?)</a>', cells[3], re.S)]
            lec["readings"] = items or [text(cells[3]).removeprefix("Readings :").strip()]
        lectures.append(lec)
    return lectures


def parse_stream(html: str) -> list[dict]:
    """Schedule as a flat stream of blocks: '<date> Lecture <n> : <title> [Slides] [Video]'."""
    body = re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S)
    start = body.find(">Schedule<")
    body = body[start:] if start >= 0 else body
    tokens: list[tuple[str, str]] = []
    for m in re.finditer(r'<a [^>]*href="([^"]+)"|>([^<>]+)<', body):
        if m.group(1):
            tokens.append(("link", m.group(1)))
        elif (t := H.unescape(m.group(2)).strip()):
            tokens.append(("text", t))

    lectures, date, section, cur = [], "", "", None
    i = 0
    while i < len(tokens):
        kind, val = tokens[i]
        if kind == "link":
            if cur is not None:
                classify(val, cur)
        elif MONTH_DATE.match(val):
            date = val
        elif val.startswith("Chapter"):
            section = val
        elif val == "Lecture" and i + 3 < len(tokens) and tokens[i + 1][1].isdigit() and tokens[i + 2][1] == ":":
            title = tokens[i + 3][1]
            if title.startswith("Chapter"):  # chapter banner rendered like a lecture row
                section, cur = title, None
                i += 4
                continue
            if NOT_A_LECTURE.search(title):
                cur = None
                i += 4
                continue
            cur = {"n": int(tokens[i + 1][1]), "date": date, "title": title,
                   "description": "", "section": section}
            lectures.append(cur)
            i += 4
            continue
        i += 1
    return lectures


def scrape(url: str) -> dict:
    html = fetch(url)
    title = text(re.search(r"<title>(.*?)</title>", html, re.S | re.I).group(1)) if "<title" in html else url
    term = re.search(r"(?:Fall|Autumn|Winter|Spring|Summer)\s+20\d\d", title + " " + text(html[:20000]))
    lectures = parse_table(html) or parse_stream(html)
    for n, lec in enumerate(lectures, 1):
        lec.setdefault("n", n)
    return {"url": url, "title": title, "term": term.group(0) if term else "", "lectures": lectures}


# ------------------------------------------------------------------ render

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def lesson_for(lec: dict, processed: dict) -> str | None:
    for key in (video_key(lec["video"]) if lec.get("video") else None,
                direct_pdf(lec["slides"]) if lec.get("slides") else None):
        if key and key in processed:
            return processed[key]["lesson"]
    return None


def render(src: dict, course: dict, processed: dict) -> str:
    lecs = course["lectures"]
    done = sum(1 for l in lecs if lesson_for(l, processed))
    out = [
        "---",
        f"title: {json.dumps(src.get('name') or course['title'])}",
        f"source: {course['url']}",
        "tags: [course]",
        "---",
        "",
        "<!-- Generated by pipeline/courses.py from the course website. Do not edit by hand. -->",
        "",
        f"> [!info] {course['title']}" + (f" · {course['term']}" if course["term"] else ""),
        f"> Course site: {course['url']}  ",
        f"> **{len(lecs)} lectures · {done} lessons written.** Lessons are added daily, in lecture order.",
        "",
    ]
    section = None
    for lec in lecs:
        if lec.get("section") and lec["section"] != section:
            section = lec["section"]
            out += [f"## {section}", ""]
        out.append(f"### Lecture {lec['n']}: {lec['title']}")
        if lec.get("date"):
            out.append(f"*{lec['date']}*")
        out.append("")
        if lec.get("description"):
            out += [lec["description"], ""]
        links = [f"[{k.title()}]({lec[k]})" for k in ("slides", "video", "lab") if lec.get(k)]
        if links:
            out += [" · ".join(links), ""]
        if lec.get("readings"):
            out += ["**Readings:**"] + [f"- {r}" for r in lec["readings"]] + [""]
        lesson = lesson_for(lec, processed)
        out += [f"**Lesson:** [[{lesson}]]" if lesson else "**Lesson:** *not written yet*", ""]
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(SOURCES.read_text()) or {}
    processed = load_state()["processed"]
    OUT.mkdir(parents=True, exist_ok=True)
    hubs, failed = [], 0
    for src in cfg.get("sources") or []:
        if src.get("type") != "course_page":
            continue
        name = src.get("name") or src["url"]
        slug = src.get("slug") or slugify(name.split("(")[0])
        try:
            course = scrape(src["url"])
        except Exception as e:
            print(f"[courses] {name}: {e} (keeping the existing hub page)", file=sys.stderr)
            failed += 1
            if (OUT / f"{slug}.md").exists():
                hubs.append((slug, name))
            continue
        if not course["lectures"]:
            print(f"[courses] {name}: no lectures parsed (page layout changed?)", file=sys.stderr)
            failed += 1
        if args.json:
            (WORK / "courses").mkdir(parents=True, exist_ok=True)
            (WORK / "courses" / f"{slug}.json").write_text(json.dumps(course, indent=2, ensure_ascii=False))
        (OUT / f"{slug}.md").write_text(render(src, course, processed))
        hubs.append((slug, name))
        print(f"[courses] {name}: {len(course['lectures'])} lectures -> content/courses/{slug}.md")

    index = ["---", "title: Courses", "---", "",
             "<!-- Generated by pipeline/courses.py. Do not edit by hand. -->", "",
             "Courses I'm following. Each page lists every lecture with its slides, video and readings, and links the lesson once it's written.", ""]
    index += [f"- [[courses/{slug}|{name}]]" for slug, name in hubs]
    (OUT / "index.md").write_text("\n".join(index) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
