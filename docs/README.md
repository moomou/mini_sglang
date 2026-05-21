# mini_sglang lesson site

Static HTML for the curriculum (L0–L3 so far). Hostable as-is on GitHub Pages.

## Local preview

```sh
cd mini_sglang/docs
python -m http.server 8000
# browse http://localhost:8000
```

## GitHub Pages setup

Two options:

**A. Host from `/docs` on the default branch**
1. Push the repo.
2. Repo → **Settings** → **Pages** → **Source**: Deploy from a branch.
3. Branch: `main` (or whatever) · Folder: `/mini_sglang/docs` (only works at repo root; if your Pages source has to be `/docs`, copy or symlink this folder there).

**B. Host from a `gh-pages` branch**
```sh
git subtree push --prefix mini_sglang/docs origin gh-pages
```
Then in Settings → Pages, choose `gh-pages` / `/`.

## Files

- `index.html` — landing page with lesson grid.
- `L0-architecture.html` — engine pipeline + module map.
- `L1-model-weights.html` — Qwen3-8B from spec, RoPE, RMSNorm, KV-cache theory.
- `L2-paged-kv.html` — paged pool + block allocator + ForwardMeta.
- `L3-paged-attention.html` — block-table-aware kernel, three metadata tensors.
- `style.css` — single stylesheet, no external deps.

## Adding a new lesson

1. Copy `L3-paged-attention.html` as `LN-<slug>.html`.
2. Add a card to `index.html`'s `.lesson-grid`.
3. Update the `<nav>` in every `topbar`.
4. Update `lesson-nav` prev/next arrows on neighboring lessons.

Each lesson page should include: 3.x section titles, real Q&A blockquotes from the session, and a "Pitfalls" / "Debug log" recap of bugs we actually hit.
