#!/usr/bin/env python3
r"""
trace_modules.py

A tracing script that measures which Python modules the similarity search server
actually uses, so that only those top-level packages are bundled in the
redistributable package.

Run the real HOOPS AI calls the server makes inside run_traced_workload()
(shape embedding generation, similarity-search indexing and querying via
FaissVectorStore and CADSearch, FAISS KMeans clustering for the Shape Map,
and thumbnail/stream-cache export), then extract the
difference in top-level package names added to sys.modules before and after the
workload runs. Each call runs in an isolated block so that one failure does not
drop the modules the other blocks would import; failures are reported on stderr.

Usage:
    export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1
    export HOOPS_AI_LICENSE='...'
    export DISPLAY=:99   # In headless environments, make sure Xvfb is running
    python3 tools/trace_modules.py \
        --cad-file /path/to/sample.step \
        --embed-ckpt /path/to/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt \
        --out /tmp/used_top_level_modules.txt

The site-packages path is built automatically from $HOOPS_AI_HOME
(Linux: $HOOPS_AI_HOME/.venv/lib/python3.12/site-packages,
 Windows: %HOOPS_AI_HOME%\.venv\Lib\site-packages).

If you change which HOOPS AI features the server exercises, rewrite
run_traced_workload() to call the actual APIs, rerun this trace, and rebuild the
package so python_packages/ stays in sync with what the server imports.

Note on the cross-references below: the block comments name the owning module
and function (``<module>.<function>()``), never a core.py line number. core.py
is now a thin facade; the logic lives in the domain modules (index_registry,
embeddings, search, context, shape_map, viewer, licensing). Line numbers drift
on every edit, so reference by module and symbol and grep for them.
"""
import argparse
import os
import sys


STDLIB_MODULE_NAMES = (
    set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
)


def resolve_site_packages() -> str:
    """Build the venv site-packages path for a standard HOOPS AI installation
    from HOOPS_AI_HOME."""
    home = os.environ.get("HOOPS_AI_HOME", "")
    if not home:
        return ""

    if os.name == "nt":
        return os.path.join(home, ".venv", "Lib", "site-packages")
    return os.path.join(home, ".venv", "lib", "python3.12", "site-packages")


def top_level_name(module_name: str) -> str:
    return module_name.split(".")[0]


def _trace_block(label: str, fn) -> None:
    """Run one workload block in isolation.

    A block that raises must not abort the remaining blocks: if it did, the
    modules the later blocks would import silently drop out of the trace and the
    redistributable package quietly shrinks. Failures are therefore caught here
    and reported on stderr (one line) rather than propagating -- a missing block
    is loud, never silent.
    """
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - keep tracing the remaining blocks
        print(f"[trace] block '{label}' failed: {exc!r}", file=sys.stderr)


def run_traced_workload(cad_file: str, embed_ckpt: str) -> None:
    """Call the HOOPS AI features the similarity search server actually uses.

    Every feature runs in its own isolated block (see :func:`_trace_block`) so a
    single failure does not hide the modules the other blocks import. Rewrite
    this to match the server's real use case. The calls here were verified
    against hoops_ai 1.1.0 and the server modules (index_registry, embeddings,
    search, context, shape_map, viewer, licensing) and vendor/; if the
    version changes, import paths or arguments may change, so verify them.
    """
    import tempfile

    import numpy as np
    import hoops_ai  # noqa: F401

    state: dict = {}

    def block_license() -> None:
        # licensing.init_hoops_license(): hoops_ai.set_license(key, validate=True).
        license_key = os.environ.get("HOOPS_AI_LICENSE", "")
        if license_key:
            hoops_ai.set_license(license_key, validate=True)

    def block_loader() -> None:
        # viewer.export_scs_for_part(): HOOPSLoader() is constructed wherever CAD is read/exported.
        from hoops_ai.cadaccess import HOOPSLoader

        state["loader"] = HOOPSLoader()

    def block_embeddings() -> None:
        # embeddings.get_embedder(): HOOPSEmbeddings.register_model(...) then
        # HOOPSEmbeddings(...); embed_shape() drives the indexing embed path.
        if not embed_ckpt:
            return
        from hoops_ai.ml.embeddings import HOOPSEmbeddings

        HOOPSEmbeddings.register_model(
            model_name="traced_model", checkpoint_path=embed_ckpt
        )
        embedder = HOOPSEmbeddings(cad_loader=state.get("loader"), model="traced_model")
        state["embedder"] = embedder
        state["raw_embeddings"] = embedder.embed_shape(cad_file)

    def block_vector_store() -> None:
        # index_registry.create_index() / index_registry._load_named_index(): FaissVectorStore
        # upsert / save / load / query.
        raw = state.get("raw_embeddings")
        if not raw:
            return
        from hoops_ai.ml.embeddings import Embedding, FaissVectorStore, VectorRecord

        vectors = [np.asarray(e.values, dtype="float32") for e in raw]
        vec = vectors[0] if len(vectors) == 1 else np.mean(vectors, axis=0)
        dim = int(vec.shape[0])
        index_base = os.path.join(state["tmp_dir"], "traced_index")

        vs = FaissVectorStore(dim)
        emb = Embedding(values=vec, model="traced_model", dim=dim)
        rec = VectorRecord(id=cad_file, embedding=emb, metadata={"file_id": cad_file})
        vs.upsert([rec])
        vs.save(index_base)

        vs = FaissVectorStore.load(index_base)
        vs.query(vec, top_k=1)

        state["vec"] = vec
        state["index_base"] = index_base
        state["vs"] = vs

    def block_cad_search() -> None:
        # index_registry._get_named_searcher(): CADSearch(
        # shape_model=embedder), load_shape_index(), search_by_shape(). This is
        # the core of the similarity search and must run past import.
        embedder = state.get("embedder")
        index_base = state.get("index_base")
        if embedder is None or index_base is None:
            return
        try:
            from hoops_ai.ml.embeddings import CADSearch
        except ImportError:  # matches index_registry._get_named_searcher()'s fallback import path
            from hoops_ai.ml import CADSearch

        searcher = CADSearch(shape_model=embedder)  # index_registry._get_named_searcher()
        # index_registry._load_shape_index_pickle_compat(): load the .faiss saved above.
        searcher.load_shape_index(path=index_base + ".faiss")
        # search.search_index(): searcher.search_by_shape(query_path, top_k=...).
        state["search_results"] = searcher.search_by_shape(cad_file, top_k=1)

    def block_context_layer() -> None:
        # context.predict_context(): build a ContextPredictor over the index's
        # context sidecar (context_store.py) and call infer() once. This imports
        # hoops_ai.ml.context_layer (and its matplotlib/pandas dependencies) so
        # the redistributable package includes the module the /predict-context
        # endpoint needs. Seeding uses the real VectorHit objects from the search
        # above so the ids line up with the provider.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import pathlib

        import context_store

        hits = [
            h
            for body in (state.get("search_results") or [])
            for h in (body or [])
            if h is not None
        ]
        if not hits:
            # Nothing to infer, but still import the module so it is traced.
            import hoops_ai.ml.context_layer  # noqa: F401
            return

        with tempfile.TemporaryDirectory() as ctx_dir:
            index_dir = pathlib.Path(ctx_dir)
            for i, h in enumerate(hits):
                hid = str(getattr(h, "id", "") or f"hit{i}")
                context_store.set_part_context(
                    index_dir,
                    hid,
                    {"Material": "Steel", "UnitCost": 10.0 + i},
                    source="trace",
                )
            provider = context_store.make_index_context_provider(index_dir)
            predictor = context_store.build_context_predictor(index_dir, provider)
            predictor.infer(hits, ["Material", "UnitCost"])

    def block_index_stats() -> None:
        # index_registry.get_index_stats() / _compute_faiss_stats(): faiss.vector_to_array(vs._index.id_map).
        vs = state.get("vs")
        if vs is None:
            return
        import faiss

        faiss.vector_to_array(vs._index.id_map)

    def block_kmeans() -> None:
        # shape_map._shape_map_cluster() clustering / vendor/assembly_matcher.py:
        # faiss.Kmeans(dim, k, ...).train(X) then km.index.search(X, 1).
        import faiss

        X = np.ascontiguousarray(np.random.RandomState(42).rand(32, 8).astype("float32"))
        km = faiss.Kmeans(X.shape[1], 4, niter=5, seed=42, verbose=False)
        km.train(X)
        km.index.search(X, 1)

    def block_stream_cache() -> None:
        # viewer.export_scs_for_part(): HOOPSTools.
        # exportStreamCache(model, ...) renders thumbnails/<id>.png and stores
        # scs/<id>.scs.
        loader = state.get("loader")
        if loader is None:
            return
        from hoops_ai.cadaccess import HOOPSTools

        model = loader.create_from_file(cad_file)
        tools = HOOPSTools()
        tools.exportStreamCache(
            model,
            filename=os.path.join(state["tmp_dir"], "traced_thumb.scs"),
            is_white_background=True,
            overwrite=True,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        state["tmp_dir"] = tmp_dir
        for label, fn in (
            ("license", block_license),
            ("loader", block_loader),
            ("embeddings", block_embeddings),
            ("vector_store", block_vector_store),
            ("cad_search", block_cad_search),
            ("context_layer", block_context_layer),
            ("index_stats", block_index_stats),
            ("kmeans", block_kmeans),
            ("stream_cache", block_stream_cache),
        ):
            _trace_block(label, fn)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-file", required=True)
    parser.add_argument("--embed-ckpt", default="")
    parser.add_argument("--out", default="/tmp/used_top_level_modules.txt")
    args = parser.parse_args()

    site_packages = resolve_site_packages()
    if site_packages:
        sys.path.insert(0, site_packages)

    before = set(sys.modules.keys())
    run_traced_workload(args.cad_file, args.embed_ckpt)
    after = set(sys.modules.keys())

    new_modules = after - before
    top_level = sorted(
        {
            top_level_name(m)
            for m in new_modules
            if top_level_name(m) not in STDLIB_MODULE_NAMES
        }
    )

    with open(args.out, "w") as f:
        for name in top_level:
            f.write(name + "\n")

    print(f"Newly imported top-level packages: {len(top_level)}")
    print(f"Wrote package list to: {args.out}")
    print("(Use this list to pass only the required directories/files from")
    print(" site-packages to tools/build_redist_package.ps1/.sh)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
