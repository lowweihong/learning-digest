# learning-digest

A daily pipeline that turns videos into first-principles lessons, a linked concept library and review questions. The site is published with Quartz v4 on GitHub Pages.

- `pipeline/digest.py`: discovers new videos, fetches transcripts, tracks state. Deterministic, no LLM.
- **You** (the Claude session): write the lessons and concept notes, following the spec below.
- `pipeline/validate.py`: structural checks. It must pass before you commit.
- `.github/workflows/deploy.yml`: builds Quartz and deploys to Pages on every push to `main`.

## Daily run (what the scheduled task does)

```bash
pip install -q -r pipeline/requirements.txt
python3 pipeline/digest.py discover          # writes work/queue.json
```

For each item in `work/queue.json` → `queue`:

1. Run `python3 pipeline/digest.py transcript <url>`. This writes `work/<id>/transcript.md` and `meta.json`. If it fails, note the error in the run summary, don't mark the video done, and move on.
2. Read the **whole** transcript. Write the lesson and the concept notes as specified below.
3. Update `content/review/index.md`: add the lesson's questions under a dated heading, newest first.
4. Run `python3 pipeline/digest.py done <id> lessons/<file-stem>`.

Then:

```bash
python3 pipeline/courses.py --json  # regenerate content/courses/* hubs (they link the new lessons)
python3 pipeline/validate.py        # fix every error before continuing
npx quartz build                    # must succeed
git add content state sources.yaml && git commit -m "digest: <date> — <n> lesson(s)" && git push origin main
```

If the queue is empty, still commit `state/` if it changed, then stop.

## Ground rules

- **Only teach what the source supports.** Each claim in a lesson comes from the transcript, or is standard background knowledge that you clearly mark as *Background*. If the transcript is thin, garbled or auto-translated, say so at the top of the lesson. Don't fill the gaps with guesses.
- Give timestamps (`[mm:ss]` links, `<url>&t=<seconds>s`) for the video's key claims, in Source notes. For a **slides** source (`kind: slides` in the transcript header), cite `[slide N]` instead, and set `duration: "<n> slides"`. Slides are terse, so mark anything you fill in between bullet points as *Background*.
- If the video makes a dubious or contested claim, flag it in a `> [!warning]` callout. Don't drop it silently.
- Write for a smart reader who is new to the topic. Be concise and concrete, with no filler or hype.

## Lesson: `content/lessons/YYYY-MM-DD-<slug>.md`

`YYYY-MM-DD` is the date the lesson is written. `<slug>` is a short kebab-case form of the topic, not the video's clickbait title.

```markdown
---
title: "<Plain-language topic title>"
source: <video url>
channel: <channel name>
date: YYYY-MM-DD           # date written
published: YYYY-MM-DD      # video upload date
duration: "mm:ss"
concepts: [Concept Name, Another Concept]   # every atomic concept introduced; each needs a note
tags: [lesson, <domain>, <subdomain>]
course: courses/<slug>    # only when the item came from a course_page source (its `source` in queue.json)
---

> [!abstract] One-sentence thesis of the video.

For a course lesson, the next line is `Part of [[courses/<slug>|<course name>]], lecture <n>.` The lecture number and title are in `work/courses/<slug>.json` (written by `courses.py --json`).

## TL;DR
- Exactly 3 bullets: the 3 things worth remembering.

## Concept map
The atomic concepts in **dependency order**: each one depends only on those above it.
1. [[concepts/<slug>|Concept Name]]: one-line definition.
   *Needs:* nothing / [[concepts/...]]
2. ...

Add a small `mermaid` graph TD of the dependencies when there are 4 or more concepts.

## First-principles rebuild
Tell the story of the idea from the ground up, organized into clear narrative acts:

### 0. The Reality Check & Motivation
The fundamental physical, algorithmic, or systems dilemma that made this work necessary. What was broken, painful, or bottlenecked before this idea?

### 1. Ground Zero (The Core Basis Concept)
Strip away library and framework jargon. Start with the bare-metal primitives and fundamental truths everyone agrees on. Explain how the domain actually works before jumping into solutions.

### 2. The Idea Journey (The Breakthrough)
Walk through how a practitioner or researcher discovers the solution:
- **The naive first attempt:** What is the obvious thing to try, and why does it fall short?
- **The conceptual leap:** The core insight or reformulation that breaks through the barrier.
- **The concrete mechanism:** How it actually executes (with concrete numbers, real lecture matrices/code, and small diagrams where helpful).
- **The honest trade-off:** What is traded away (accuracy, compute, latency, flexibility), and when this method should *not* be used.
Embed timestamp or slide citations `[mm:ss]` / `[slide N]` naturally into the narrative prose.

### Putting it together
How all the atomic concepts combine into the final working system or result.

## Connections
- Links to existing `content/concepts/*` notes from earlier lessons that this builds on, contrasts with or generalizes. Check the existing library first, with `ls content/concepts` and grep.
- Open questions, or what to learn next.

## Review questions
At least 4, ordered from easy to hard. Include at least one of each type: **recall**, **explain-why**, **apply**, **transfer** (use the idea in a new domain).
> [!question]- Q1 (recall): <question>
> <answer, 1–4 sentences, linking the concept note>

## Source notes
- [mm:ss](<url>&t=<s>s): key claim or moment
```

## Concept note: `content/concepts/<slug>.md`

The slug is the lowercase kebab-case form of the concept name, exactly as it appears in the lesson's `concepts:` list. For example, "Gradient Descent" becomes `gradient-descent`.

**Atomic** means one idea per note. A note that needs "and" in its title is two notes.

**Check before you create.** Look for an existing note, including its `aliases`. If one exists, **merge into it**: add the new lesson under "Appears in", and improve the definition or example only if the new source adds something. Never create duplicates or near-duplicates.

```markdown
---
title: Concept Name
aliases: [abbreviation, common synonym]
tags: [concept, <domain>]
---

## Definition
One or two sentences, precise enough to be checked.

## Why it's true
The first-principles justification in a few lines.

## Minimal example
The smallest example that makes it click.

## Prerequisites
- [[concepts/...]], or "None"

## Common confusions
- Optional. Leave the section out if you have nothing real to say.

## Appears in
- [[lessons/YYYY-MM-DD-slug]]: how this lesson uses it.
```

## Conventions

- Wikilinks are path-style and have no `.md`: `[[concepts/gradient-descent|gradient descent]]`.
- Quartz renders Obsidian callouts, mermaid, KaTeX (`$...$`) and backlinks.
- Don't edit `quartz/` internals for content work. Site config lives in `quartz.config.ts` and `quartz.layout.ts`.
- The add-a-source format is documented at the top of `sources.yaml`.
- `content/courses/*.md` are generated by `pipeline/courses.py`. Don't edit them by hand: re-run the script.
