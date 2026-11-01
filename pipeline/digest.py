#!/usr/bin/env python3
"""Deterministic half of the learning-digest pipeline.

The lesson writing itself is done by the scheduled Claude session (see CLAUDE.md).
This script only discovers new videos, fetches transcripts, and tracks state.

Commands:
  discover [--dry-run]       Scan sources.yaml, print new videos as JSON and write work/queue.json.
  transcript <url-or-id>     Fetch metadata + timestamped transcript into work/<id>/.
  done <id> <lesson-path>    Record a video as processed (state/processed.json).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "sources.yaml"
STATE = ROOT / "state" / "processed.json"
WORK = ROOT / "work"

YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")


def log(*a):
    print(*a, file=sys.stderr)


def load_state() -> dict:
    if STATE.exists():
        s = json.loads(STATE.read_text())
    else:
        s = {}
    s.setdefault("processed", {})
    s.setdefault("seen_sources", {})
    s.setdefault("skipped", {})
    return s


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")


def video_key(url: str) -> str:
    """Stable ID: the 11-char YouTube ID when there is one, else the URL."""
    m = YT_ID.search(url)
    return m.group(1) if m else url


def ydl_opts(**extra) -> dict:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    cookies = os.environ.get("YTDLP_COOKIES")
    if cookies and Path(cookies).exists():
        opts["cookiefile"] = cookies
    opts.update(extra)
    return opts


# ---------------------------------------------------------------- discovery

def list_youtube(url: str, limit: int) -> list[dict]:
    """Newest-first entries from a channel or playlist (flat: no per-video requests)."""
    import yt_dlp

    if "/playlist" not in url and "list=" not in url:
        url = url.rstrip("/")
        if not re.search(r"/(videos|streams|shorts)$", url):
            url += "/videos"
    with yt_dlp.YoutubeDL(ydl_opts(extract_flat="in_playlist", playlistend=limit)) as ydl:
        info = ydl.extract_info(url, download=False)
    out = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        out.append({
            "id": e["id"],
            "url": e.get("url") if str(e.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={e['id']}",
            "title": e.get("title"),
            "channel": info.get("channel") or info.get("uploader") or info.get("title"),
        })
    return out


def list_rss(url: str, limit: int) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "learning-digest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    items = []
    for it in root.iter("item"):  # RSS 2.0
        link = (it.findtext("link") or "").strip()
        items.append({"url": link, "title": (it.findtext("title") or "").strip()})
    for it in root.iter("{http://www.w3.org/2005/Atom}entry"):  # Atom (incl. YouTube feeds)
        l = it.find("a:link", ns)
        link = l.get("href") if l is not None else ""
        items.append({"url": link, "title": (it.findtext("a:title", default="", namespaces=ns)).strip()})
    feed_title = root.findtext("channel/title") or root.findtext("a:title", default="", namespaces=ns)
    out = []
    for i in items[:limit]:
        if i["url"]:
            out.append({"id": video_key(i["url"]), "url": i["url"], "title": i["title"], "channel": feed_title})
    return out


PLAYLIST = re.compile(r"list=([A-Za-z0-9_-]{13,})")


def list_course_page(url: str) -> list[dict]:
    """All YouTube videos linked from a course page, in page (i.e. lecture) order.

    Playlists linked from the page are expanded in place. Login-only hosts
    (Panopto, Canvas) can't be read and are ignored.
    """
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 learning-digest/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        html = r.read().decode("utf-8", errors="ignore")
    title = (re.search(r"<title>(.*?)</title>", html, re.S | re.I) or [None, url])[1].strip()
    out, seen = [], set()
    for m in re.finditer(r"https?://(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/[^\s\"'<>]+", html):
        link = m.group(0).replace("&amp;", "&")
        vid = YT_ID.search(link)
        if vid:
            items = [{"id": vid.group(1), "url": f"https://www.youtube.com/watch?v={vid.group(1)}", "title": None}]
        elif pl := PLAYLIST.search(link):
            items = list_youtube(f"https://www.youtube.com/playlist?list={pl.group(1)}", 500)
        else:
            continue
        for e in items:
            if e["id"] not in seen:
                seen.add(e["id"])
                out.append({**e, "channel": title})
    return out


def cmd_discover(args) -> None:
    cfg = yaml.safe_load(SOURCES.read_text()) or {}
    max_per_run = int(cfg.get("max_per_run", 3))
    backfill = int(cfg.get("initial_backfill", 1))
    scan = int(cfg.get("scan_depth", 15))
    state = load_state()
    done = set(state["processed"]) | set(state["skipped"])
    queue, errors = [], []

    for src in cfg.get("sources") or []:
        kind, url = src.get("type"), src.get("url")
        name = src.get("name") or url
        try:
            if kind == "video":
                entries = [{"id": video_key(url), "url": url, "title": None, "channel": None}]
            elif kind in ("youtube_channel", "youtube_playlist"):
                entries = list_youtube(url, scan)
            elif kind == "rss":
                entries = list_rss(url, scan)
            elif kind == "course_page":
                entries = list_course_page(url)
            else:
                raise ValueError(f"unknown source type {kind!r}")
        except Exception as e:  # one broken source must not stop the run
            errors.append({"source": name, "error": str(e)[:300]})
            log(f"[discover] {name}: {e}")
            continue

        # Courses are worked through from lecture 1 (max_per_run per day), so no backfill cut.
        first_scan = url not in state["seen_sources"] and kind not in ("video", "course_page")
        new = [e for e in entries if e["id"] not in done]
        if first_scan:
            # Don't flood the digest with a channel's back catalogue: take the newest few,
            # mark the rest as skipped so they never come back.
            for e in new[backfill:]:
                state["skipped"][e["id"]] = {"reason": "pre-existing on first scan", "source": name}
            new = new[:backfill]
        for e in new:
            e["source"] = name
        queue.extend(new)
        if not args.dry_run:
            state["seen_sources"][url] = dt.date.today().isoformat()

    # de-dupe across sources, cap per run
    seen, capped = set(), []
    for e in queue:
        if e["id"] not in seen:
            seen.add(e["id"])
            capped.append(e)
    result = {"queue": capped[:max_per_run], "deferred": len(capped) - min(len(capped), max_per_run), "errors": errors}
    if not args.dry_run:
        save_state(state)
        WORK.mkdir(exist_ok=True)
        (WORK / "queue.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# --------------------------------------------------------------- transcripts

def fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def group_segments(segs: list[tuple[float, str]], every: float = 30.0) -> list[str]:
    """Merge caption fragments into ~30s paragraphs, each prefixed with its start time."""
    lines, buf, start = [], [], None
    for t, text in segs:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if start is None:
            start = t
        if buf and buf[-1] == text:  # auto-caption duplicates
            continue
        buf.append(text)
        if t - start >= every:
            lines.append(f"[{fmt_ts(start)}] " + " ".join(buf))
            buf, start = [], None
    if buf:
        lines.append(f"[{fmt_ts(start)}] " + " ".join(buf))
    return lines


def transcript_api(video_id: str) -> list[tuple[float, str]] | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        api = YouTubeTranscriptApi()
        tl = api.list(video_id)
        try:
            t = tl.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except Exception:
            t = tl.find_transcript(["en", "en-US", "en-GB"])
        return [(s.start, s.text) for s in t.fetch()]
    except Exception as e:
        log(f"[transcript] youtube-transcript-api failed: {e}")
        return None


def parse_vtt(text: str) -> list[tuple[float, str]]:
    segs, t = [], None
    for line in text.splitlines():
        m = re.match(r"(\d+):(\d+):(\d+)\.(\d+) -->", line) or re.match(r"(\d+):(\d+)\.(\d+) -->", line)
        if m:
            g = [int(x) for x in m.groups()]
            t = g[0] * 3600 + g[1] * 60 + g[2] if len(g) == 4 else g[0] * 60 + g[1]
            continue
        if t is not None and line.strip() and "-->" not in line and not line.startswith(("WEBVTT", "Kind:", "Language:")):
            segs.append((t, re.sub(r"<[^>]+>", "", line)))
    return segs


def cmd_transcript(args) -> None:
    import yt_dlp

    url = args.video if args.video.startswith("http") else f"https://www.youtube.com/watch?v={args.video}"
    vid = video_key(url)
    out = WORK / re.sub(r"[^A-Za-z0-9_-]", "_", vid)[:80]
    out.mkdir(parents=True, exist_ok=True)

    if args.from_file:  # manual fallback: a pasted transcript, when the site can't be reached
        text = Path(args.from_file).read_text()
        meta = {"id": vid, "title": args.title or vid, "webpage_url": url, "duration": 0}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        (out / "transcript.md").write_text(
            f"# {meta['title']}\n\nurl: {url}\ntranscript-source: pasted ({args.from_file})\n\n## Transcript\n{text}\n")
        print(out / "transcript.md")
        return

    try:
        with yt_dlp.YoutubeDL(ydl_opts(
            writesubtitles=True, writeautomaticsub=True, subtitleslangs=["en.*", "en"],
            subtitlesformat="vtt", outtmpl=str(out / "subs.%(ext)s"),
        )) as ydl:
            info = ydl.extract_info(url, download=True)  # skip_download: only subs are written
    except yt_dlp.utils.DownloadError as e:
        hint = ""
        if "403" in str(e) and "proxy" in str(e).lower():
            hint = "\nHint: the network policy blocks this host (allow youtube.com, youtu.be, googlevideo.com)."
        elif "Sign in to confirm" in str(e):
            hint = "\nHint: YouTube is bot-checking this IP; set YTDLP_COOKIES to a cookies.txt file."
        sys.exit(f"transcript fetch failed for {url}: {str(e)[:300]}{hint}")

    meta = {k: info.get(k) for k in (
        "id", "title", "channel", "uploader", "upload_date", "duration", "webpage_url", "description", "tags", "chapters",
    )}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    segs = transcript_api(vid) if len(vid) == 11 else None
    source = "youtube-transcript-api"
    if not segs:
        vtts = sorted(out.glob("subs*.vtt"), key=lambda p: (".en." not in p.name, len(p.name)))
        if vtts:
            segs, source = parse_vtt(vtts[0].read_text(errors="ignore")), f"yt-dlp:{vtts[0].name}"
    if not segs:
        sys.exit(f"No transcript available for {url}")

    chapters = "".join(f"- [{fmt_ts(c['start_time'])}] {c['title']}\n" for c in meta.get("chapters") or [])
    body = (
        f"# {meta['title']}\n\nchannel: {meta.get('channel') or meta.get('uploader')}\n"
        f"url: {meta.get('webpage_url')}\nuploaded: {meta.get('upload_date')}\n"
        f"duration: {fmt_ts(meta.get('duration') or 0)}\ntranscript-source: {source}\n\n"
        + (f"## Chapters\n{chapters}\n" if chapters else "")
        + "## Transcript\n" + "\n".join(group_segments(segs)) + "\n"
    )
    (out / "transcript.md").write_text(body)
    print(out / "transcript.md")


def cmd_done(args) -> None:
    state = load_state()
    state["processed"][args.id] = {"lesson": args.lesson, "date": dt.date.today().isoformat()}
    save_state(state)
    print(f"recorded {args.id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover"); d.add_argument("--dry-run", action="store_true"); d.set_defaults(fn=cmd_discover)
    t = sub.add_parser("transcript"); t.add_argument("video")
    t.add_argument("--from-file", help="use a pasted transcript instead of fetching"); t.add_argument("--title"); t.set_defaults(fn=cmd_transcript)
    k = sub.add_parser("done"); k.add_argument("id"); k.add_argument("lesson"); k.set_defaults(fn=cmd_done)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
