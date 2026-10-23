<!-- Keep this short. Tick every box, or explain in the PR body. -->

- [ ] If `web/src` changed: ran `npm run build` and confirmed it succeeds (`static/app` is a build output and is not committed).
- [ ] Search scores and ranking unchanged (or the diff from the baseline is noted below). Baseline: 304.stp / top_k=301 (warm) -> 311.stp = 0.7311, 97 total hits.
- [ ] No response-schema fields removed or renamed (additions are fine).
- [ ] No license keys or real data committed (`.env`, `uploads/`, `indexes/`, `thumbnails/`, `scs/`).
- [ ] `pytest` result (count): <!-- e.g. 608 passed, 9 skipped -->
