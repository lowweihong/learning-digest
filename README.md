# learning-digest

A daily pipeline that does three things:
- fetches new videos from the sources in `sources.yaml`
- breaks each one into atomic concepts and rebuilds it from first principles
- publishes the lessons, a linked concept library and review questions as a [Quartz](https://quartz.jzhao.xyz) site on GitHub Pages

| Piece | What it does |
|---|---|
| `sources.yaml` | Channels, playlists, RSS feeds and single videos to follow |
| `pipeline/digest.py` | `discover` finds new videos, `transcript` fetches metadata and a timestamped transcript, `done` records a finished video |
| `CLAUDE.md` | The lesson and concept spec the daily Claude session follows |
| `pipeline/validate.py` | Structural checks: sections, at least 4 review questions, every concept has a note, no broken wikilinks |
| `content/` | The site: `lessons/`, `concepts/`, `review/` |
| `.github/workflows/deploy.yml` | Builds Quartz and deploys to Pages on every push to `main` |

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r pipeline/requirements.txt && npm ci
.venv/bin/python pipeline/digest.py discover --dry-run
.venv/bin/python pipeline/digest.py transcript https://youtu.be/<id>     # or: --from-file pasted.txt --title "..."
.venv/bin/python pipeline/validate.py
npx quartz build --serve
```

## One-time setup
1. Settings → Pages → Source: **GitHub Actions**.
2. If YouTube bot-checks the runner's IP, export a `cookies.txt` and point the `YTDLP_COOKIES` environment variable at it.

Quartz's own README is kept as `QUARTZ_README.md`.
