"""Acceptance gate for the search-result cache: prove it changes nothing.

The cache added to ``search.search_index`` must return, on a hit, exactly what the
uncached path would have computed -- same hits, same scores, same order, same
count. This script proves that empirically against a real index by running each
search three ways and comparing them element by element:

    1. cache disabled  (``HOOPS_AI_SEARCH_CACHE=0``) -> the ground truth
    2. cache enabled, first call  (a cache miss that populates the entry)
    3. cache enabled, second call (a cache hit served from the entry)

All three must agree on ``[(hit["id"], round(hit["score"], 6)) for hit in hits]``
*including length and order*, or the run fails loudly and non-zero.

It also proves the tag sidecar is part of the cache key: it runs a tag-filtered
search, adds a tag, and re-runs. Because ``_write_tags_document`` never touches
the ``.faiss`` file, a cache keyed only on the faiss mtime would serve a stale
tag-filtered result here; a correct key recomputes, and the post-tag cached run
still matches its own fresh baseline.

The operator (not the agent) runs this, because it needs the HOOPS AI SDK and a
valid license. It is otherwise SDK-free -- it only imports ``core`` and drives
the same public entry points a real request uses:

    python tools/verify_search_cache.py \
        --index tmcad \
        --single-body <file_id> \
        --multi-body  <file_id> \
        --assembly    <file_id> \
        --tag-query   <file_id> \
        --tag-part    <file_id> \
        --top-k 20

Only ``--index`` and ``--single-body`` are required; the other queries are
skipped if not supplied, and each skipped scenario is reported as SKIP rather
than silently dropped. No file is written except the tag sidecar the tag test
deliberately mutates (and it restores the previous tags on exit).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import traceback

# Allow ``import core`` when run as ``python tools/verify_search_cache.py``:
# the script's own directory (tools/) is on sys.path[0], not the repo root.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import core  # noqa: E402
import search  # noqa: E402


def _fingerprint(result: dict) -> tuple:
    """The comparable essence of a search response: ordered (id, score) + count.

    Scores are rounded to 6 decimals so a float round-trip through the deep
    copy cannot register as a spurious difference; anything the cache actually
    changed would move a digit well above that.
    """
    hits = result.get("hits", [])
    ordered = tuple((h["id"], round(float(h["score"]), 6)) for h in hits)
    return (result.get("count"), ordered)


def _run(name: str, kwargs: dict, cache_on: bool) -> dict:
    if cache_on:
        os.environ["HOOPS_AI_SEARCH_CACHE"] = "1"
    else:
        os.environ["HOOPS_AI_SEARCH_CACHE"] = "0"
        core.clear_search_cache()
    return search.search_index(name, **kwargs)


def _compare_three(label: str, name: str, kwargs: dict) -> bool:
    """Run the three ways and compare. Returns True on agreement."""
    core.clear_search_cache()
    baseline = _fingerprint(_run(name, kwargs, cache_on=False))
    miss = _fingerprint(_run(name, kwargs, cache_on=True))   # populates
    hit = _fingerprint(_run(name, kwargs, cache_on=True))    # served from cache

    ok = baseline == miss == hit
    count = baseline[0]
    if ok:
        print(f"  PASS  {label}: {count} hits identical across all three runs")
        return True

    print(f"  FAIL  {label}: results diverged")
    print(f"        baseline (cache off) : count={baseline[0]} ids={[i for i, _ in baseline[1]]}")
    print(f"        cache miss (1st on)  : count={miss[0]} ids={[i for i, _ in miss[1]]}")
    print(f"        cache hit  (2nd on)  : count={hit[0]} ids={[i for i, _ in hit[1]]}")
    for idx, (b, m, h) in enumerate(zip(baseline[1], miss[1], hit[1])):
        if not (b == m == h):
            print(f"        first divergence at position {idx}: "
                  f"baseline={b} miss={m} hit={h}")
            break
    return False


def _prove_tag_etag_in_key(name: str, kwargs: dict, tag_part: str,
                           tag: str, previous_tags: list) -> bool:
    """Actually create the stale-entry state a missing ETag would expose.

    ``_compare_three`` clears the cache before every run, so it can never catch
    a stale tag-filtered entry: the bad state is "an entry cached *before* a tag
    changed is still served *after*". ``_write_tags_document`` writes only the
    JSON sidecar and never touches the ``.faiss`` file, so the searcher-cache
    key (index name + faiss mtime) is unchanged by tagging -- only the tag ETag
    in the search-cache key changes. This exercises exactly that:

      1. cache ON, run the tag-filtered search  -> seeds a cache entry
      2. add ``tag`` to ``tag_part`` WITHOUT clearing the cache
      3. cache ON, run the identical search     -> a missing ETag serves step 1
      4. cache OFF, run it again                -> the post-tag ground truth

    Step 3 must equal step 4. If the ETag is absent from the key, step 3 returns
    the pre-tag result and diverges, and this fails loudly. Use a ``tag`` that no
    part carries yet (the default is unlikely to collide): before step 2 the
    filter matches nothing, after it matches ``tag_part``, so the divergence is
    unambiguous -- provided ``tag_part`` is within the query's over-fetched
    candidate set (choose a ``--tag-part`` that appears in the unfiltered hits).
    """
    os.environ["HOOPS_AI_SEARCH_CACHE"] = "1"
    core.clear_search_cache()

    seed = _fingerprint(search.search_index(name, **kwargs))       # 1. seed entry

    new_tags = sorted(set(previous_tags) | {tag})
    core.set_part_tags(name, tag_part, new_tags)                 # 2. tag, no clear
    print("  (added tag '%s' to part '%s', cache left intact)" % (tag, tag_part))

    after_cached = _fingerprint(search.search_index(name, **kwargs))  # 3. cache ON

    os.environ["HOOPS_AI_SEARCH_CACHE"] = "0"
    core.clear_search_cache()
    truth = _fingerprint(search.search_index(name, **kwargs))         # 4. cache OFF
    os.environ["HOOPS_AI_SEARCH_CACHE"] = "1"

    if after_cached == truth:
        print("  PASS  tag-key proof: after tagging, the cached search recomputed "
              "(%d hits) and matched the post-tag baseline" % truth[0])
        if seed == truth:
            print("        note: seed and post-tag results were equal, so this run "
                  "did not force a visible change -- pick a --tag-part that appears "
                  "in the unfiltered hits to make the proof bite.")
        return True

    print("  FAIL  tag-key proof: the cached search returned a STALE tag-filtered "
          "result after a tag was added.")
    print("        The tag sidecar ETag is NOT in the cache key: tagging does not")
    print("        touch the .faiss file, so the pre-tag entry was served unchanged.")
    print("        cached-after-tag: count=%d ids=%s"
          % (after_cached[0], [i for i, _ in after_cached[1]]))
    print("        post-tag truth  : count=%d ids=%s"
          % (truth[0], [i for i, _ in truth[1]]))
    return False


def _scenarios(args) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if args.single_body:
        out.append(("single-body, kind=part",
                    dict(file_id=args.single_body, top_k=args.top_k, kind="part")))
        out.append(("single-body, kind=any, no filter",
                    dict(file_id=args.single_body, top_k=args.top_k, kind="any")))
    if args.multi_body:
        out.append(("multi-body, kind=part",
                    dict(file_id=args.multi_body, top_k=args.top_k, kind="part")))
    if args.assembly:
        out.append(("assembly query, kind=assembly",
                    dict(file_id=args.assembly, top_k=args.top_k, kind="assembly")))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Prove the search-result cache changes nothing.")
    p.add_argument("--index", required=True, help="Index name to search.")
    p.add_argument("--single-body", help="file_id of a single-body part query.")
    p.add_argument("--multi-body", help="file_id of a multi-body part query.")
    p.add_argument("--assembly", help="file_id of an assembly query.")
    p.add_argument("--tag-query", help="file_id to run for the tag-key proof.")
    p.add_argument("--tag-part", help="registered part_id to (re)tag for the tag-key proof.")
    p.add_argument("--tag", default="cache-verify-tag", help="Tag text to add in the tag-key proof.")
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    # Same as any script that drives core directly rather than through the
    # server: no lifespan has activated the SDK license for it. Without this
    # the first search fails on an inactive license before any comparison can
    # run.
    core.init_hoops_license()

    print(f"Verifying search-result cache on index '{args.index}' (top_k={args.top_k})")
    print("Each scenario runs 3x: cache off (truth), cache on miss, cache on hit.\n")

    failures = 0
    total = 0

    for label, kwargs in _scenarios(args):
        total += 1
        if not _compare_three(label, args.index, kwargs):
            failures += 1

    for req, why in (("multi_body", "multi-body query"),
                     ("assembly", "assembly query")):
        if not getattr(args, req):
            print(f"  SKIP  {why}: pass --{req.replace('_', '-')} <file_id> to include it")

    # Tag-key proof: with/without a tag filter, then add a tag and re-run.
    if args.tag_query and args.tag_part:
        print("\nTag-key proof (proves the tag sidecar is in the cache key):")
        no_filter = ("tag-query, no tag filter",
                     dict(file_id=args.tag_query, top_k=args.top_k, kind="any"))
        with_filter = ("tag-query, filtered on '%s'" % args.tag,
                       dict(file_id=args.tag_query, top_k=args.top_k, kind="any",
                            tags=[args.tag], tag_mode="any"))

        total += 2
        if not _compare_three(no_filter[0], args.index, no_filter[1]):
            failures += 1
        if not _compare_three(with_filter[0], args.index, with_filter[1]):
            failures += 1

        # Snapshot current tags so we can restore them on exit.
        previous_doc, _etag = core.get_part_tags(args.index, args.tag_part)
        previous_tags = list(previous_doc.get("tags", []))
        try:
            total += 1
            if not _prove_tag_etag_in_key(
                args.index, with_filter[1], args.tag_part, args.tag, previous_tags
            ):
                failures += 1
        finally:
            core.set_part_tags(args.index, args.tag_part, previous_tags)
            print("  (restored previous tags on part '%s')" % args.tag_part)
    else:
        print("\n  SKIP  tag-key proof: pass --tag-query <file_id> and --tag-part <part_id>")

    print(f"\n{total - failures}/{total} scenarios passed.")
    if failures:
        print("RESULT: FAIL -- the cache changed a result. Do not ship it as-is.")
        return 1
    print("RESULT: PASS -- the cache is transparent for every scenario checked.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - operator-facing script, surface the trace
        traceback.print_exc()
        sys.exit(2)
