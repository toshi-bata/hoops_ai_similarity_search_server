import io
from typing import List, Literal, Optional

import core
import search
import assembly
import shape_map
import context_store
import index_registry
import embeddings
import storage
from fastapi import APIRouter, Body, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/similarity", tags=["CAD Similarity Search"])

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Response schemas -- part similarity compare
# ---------------------------------------------------------------------------


class EmbedResponse(BaseModel):
    """Result of computing a shape embedding for a single CAD part."""

    file_id: str
    filename: str
    dim: int
    model_name: str
    num_bodies: int
    cached: bool
    vector: Optional[list[float]] = None


class CompareFileInfo(BaseModel):
    index: int
    file_id: str
    filename: str
    num_bodies: int


class ComparePair(BaseModel):
    a: int
    b: int
    score: float


class CompareError(BaseModel):
    filename: str
    detail: str


class CompareResponse(BaseModel):
    """Pairwise cosine similarity matrix for a set of CAD parts."""

    count: int
    model_name: str
    files: list[CompareFileInfo]
    matrix: list[list[float]]
    pairs: list[ComparePair]
    errors: list[CompareError]


# ---------------------------------------------------------------------------
# Limits for ZIP extraction
# ---------------------------------------------------------------------------

_ZIP_MAX_TOTAL_BYTES = core.ZIP_MAX_TOTAL_BYTES
_ZIP_MAX_FILES = core.ZIP_MAX_FILES


class _BytesUploadFile:
    """Minimal duck-typed UploadFile used to pass in-memory bytes to core helpers."""

    def __init__(self, data: bytes, filename: str) -> None:
        self.filename = filename
        self.file = io.BytesIO(data)


# ---------------------------------------------------------------------------
# POST /similarity/embed
# ---------------------------------------------------------------------------


@router.post("/embed", response_model=EmbedResponse)
def similarity_embed(
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id returned by POST /files/upload"),
    include_vector: bool = Query(
        False,
        description="When true, include the raw embedding vector in the response. "
        "Omit (default) to save bandwidth.",
    ),
):
    """Compute (or retrieve from cache) the shape embedding for a CAD part.

    Supply **either** a file upload *or* a ``file_id`` from a previous upload.
    Results are cached in memory and on disk -- repeated calls for the same file
    are fast.  Set ``include_vector=true`` to include the raw float array.

    This endpoint does **not** require a FAISS index -- the embedding model alone
    is sufficient.

    The server loads a single embedding checkpoint, resolved from
    ``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` or the ``models/`` directories.
    """
    try:
        if file_id:
            resolved_id = file_id
        elif file:
            resolved_id, _, _ = storage.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")

        result = embeddings.compute_embedding(resolved_id)
        return EmbedResponse(
            file_id=result["file_id"],
            filename=result["filename"],
            dim=result["dim"],
            model_name=result["model_name"],
            num_bodies=result["num_bodies"],
            cached=result["cached"],
            vector=[float(v) for v in result["vector"]] if include_vector else None,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/compare
# ---------------------------------------------------------------------------


@router.post("/compare", response_model=CompareResponse)
def similarity_compare(
    file_ids: Optional[str] = Query(
        None,
        description="Comma-separated file_ids of already-uploaded CAD files.",
    ),
    files: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
):
    """Compare multiple CAD parts by cosine similarity of their shape embeddings.

    Input sources can be combined freely:

    * ``file_ids`` -- comma-separated ``file_id`` values from previous uploads
    * ``files`` -- one or more multipart CAD file uploads
    * ``zip_file`` -- a single ZIP archive containing CAD files (auto-extracted)

    At least **two** valid parts are required overall.  Per-file embed failures
    are collected in ``errors`` and do not abort the whole request, unless fewer
    than two parts succeed (returns 422 in that case).

    ZIP archives are extracted safely (Zip Slip paths are rejected) and filtered
    to recognised CAD extensions only.  Uncompressed size is capped at 500 MB
    and file count at 50.

    The server loads a single embedding checkpoint, resolved from
    ``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` or the ``models/`` directories.

    This endpoint does **not** require a FAISS index.
    """
    errors: list[dict] = []
    # (file_id, display_filename) tuples collected from all input sources
    resolved: list[tuple[str, str]] = []

    # ── 1. file_ids from query parameter ─────────────────────────────────────
    if file_ids:
        for fid in [f.strip() for f in file_ids.split(",") if f.strip()]:
            try:
                path = storage.find_persistent_CAD_file(fid)
                parts = path.name.split("_", 1)
                display = parts[1] if len(parts) == 2 and len(parts[0]) == 64 else path.name
                resolved.append((fid, display))
            except RuntimeError as exc:
                errors.append({"filename": fid, "detail": str(exc)})

    # ── 2. Direct file uploads ────────────────────────────────────────────────
    if files:
        for upload in files:
            if not (upload and upload.filename):
                continue
            try:
                fid, _, _ = storage.upload_CAD_file_persistent(upload)
                resolved.append((fid, upload.filename))
            except Exception as exc:
                errors.append({"filename": upload.filename or "unknown", "detail": str(exc)})

    # ── 3. ZIP archive ────────────────────────────────────────────────────────
    if zip_file and zip_file.filename:
        try:
            zip_data = zip_file.file.read()
            _process_zip(zip_data, resolved, errors)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

    # ── Require at least 2 inputs ─────────────────────────────────────────────
    if len(resolved) < 2:
        msg = (
            f"At least 2 CAD files are required for comparison. "
            f"Received {len(resolved)} valid input(s)"
            + (f" and {len(errors)} failure(s)." if errors else ".")
        )
        raise HTTPException(status_code=422, detail=msg)

    # ── 4. Compute embeddings (collect per-file errors) ───────────────────────
    valid_ids: list[str] = []
    for fid, display in resolved:
        try:
            embeddings.compute_embedding(fid)
            valid_ids.append(fid)
        except Exception as exc:
            errors.append({"filename": display, "detail": str(exc)})

    if len(valid_ids) < 2:
        raise HTTPException(
            status_code=422,
            detail=f"At least 2 successful embeddings are required. "
            f"{len(errors)} file(s) failed to embed.",
        )

    # ── 5. Compare ────────────────────────────────────────────────────────────
    try:
        result = embeddings.compare_embeddings(valid_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {exc}") from exc

    return CompareResponse(
        count=result["count"],
        model_name=result["model_name"],
        files=[CompareFileInfo(**f) for f in result["files"]],
        matrix=result["matrix"],
        pairs=[ComparePair(**p) for p in result["pairs"]],
        errors=[CompareError(**e) for e in errors],
    )


# ---------------------------------------------------------------------------
# Response schemas -- index management
# ---------------------------------------------------------------------------


class IndexInfo(BaseModel):
    name: str
    count: Optional[int] = None
    last_modified: Optional[str] = None
    # True when this index is named in HOOPS_AI_WARMUP_INDEXES (a startup-warmup
    # target). Additive field: clients that ignore it are unaffected; the web
    # client uses it to pick a warm index as the initial selection.
    warm: bool = False


class IndexCreateResponse(BaseModel):
    name: str
    count: int
    dim: int


class IndexAddResponse(BaseModel):
    name: str
    added: int
    updated: int
    index_count: int
    errors: list[dict]


class IndexRemoveResponse(BaseModel):
    name: str
    removed: int
    index_count: int


class IndexSearchHit(BaseModel):
    id: str
    score: float
    metadata: Optional[dict] = None


class TagFilterInfo(BaseModel):
    """Why a tag-filtered search returned what it did.

    Present only when a tag filter was applied. ``matching_parts_in_index``
    against a low ``count`` says the tag exists but did not rank inside
    ``candidates_examined`` -- which otherwise looks like a bug.
    """

    tags: list[str] = []
    tag_mode: str = "any"
    tagged: Optional[bool] = None
    candidates_examined: int
    matching_parts_in_index: int


class IndexSearchResponse(BaseModel):
    hits: list[IndexSearchHit]
    count: int
    image_url: Optional[str] = None
    tag_filter: Optional[TagFilterInfo] = None


class IndexWarmResponse(BaseModel):
    status: str
    file_id: str
    count: int
    cache_enabled: bool


class AssemblySearchHit(BaseModel):
    id: str
    score: float
    geom_score: float
    coverage: float
    matched_parts: int
    candidate_parts: int
    query_parts: int
    metadata: Optional[dict] = None


class AssemblySearchResponse(BaseModel):
    hits: list[AssemblySearchHit]
    count: int
    image_url: Optional[str] = None
    tag_filter: Optional[TagFilterInfo] = None


class ShapeMapPoint(BaseModel):
    file_id: str
    filename: Optional[str] = None
    x: float
    y: float
    z: float
    cluster: int
    # Cosine similarity to this part's cluster centroid; the same scale as a
    # search score. 1.0 for parts in the Outliers group, which have no centre
    # of their own to be measured against.
    sim: Optional[float] = None
    tags: Optional[list[str]] = None


class ShapeMapLegendEntry(BaseModel):
    """One legend row: a label, its kind, how many mapped parts it covers, and
    the RGB the server baked into the point cloud for it.

    ``kind`` is ``"cluster"`` (colour-by-cluster), ``"outliers"`` (points split
    off from their cluster by ``outlier_threshold``), or ``"tag"`` /
    ``"untagged"`` (colour-by-tag). There is no overflow kind: both colourings
    give every category its own row and cycle the palette past 24. ``rgb`` is
    ``[r, g, b]`` (0-255); the client draws the swatch from it so it never
    diverges from the baked colours.

    ``cohesion`` is the mean cosine similarity of this cluster's parts to its
    own centre (1.0 = identical, lower = mixed and worth drilling into). It is
    set only for ``kind == "cluster"``; every other kind carries ``None``.

    ``centre_file_id`` is the file id of this cluster's centre part: the real
    part whose vector is closest to the cluster's k-means centroid. The centroid
    itself is the mean of the members and is not a part that exists, so this is
    the existing part chosen to stand in for it. It is NOT a k-medoids medoid,
    which is defined as the member minimising the sum of distances to all the
    others. Set only for ``kind == "cluster"``.
    """

    label: str
    kind: str
    count: int
    rgb: list[int]
    cohesion: Optional[float] = None
    centre_file_id: Optional[str] = None


class ShapeMapGroup(BaseModel):
    """One display category rendered as a single node in the point-cloud SCS.

    ``offset``/``count`` delimit this group's contiguous run in ``points`` (the
    points array is ordered by group), so a viewer maps a picked
    ``(node, index-within-node)`` to ``offset + index``. ``node_name`` is the SCS
    node name the viewer matches by name (never by creation order). ``rgb`` is the
    single ``[r, g, b]`` (0-255) baked into that node. ``key`` identifies the
    category (``"cluster:<id>"`` / ``"tag:<name>"`` / ``"outliers"`` /
    ``"untagged"``). ``cohesion`` and ``centre_file_id`` mirror
    :class:`ShapeMapLegendEntry`: set for ``cluster:<id>`` groups, ``None``
    otherwise. ``centre_file_id`` names the real part closest to that cluster's
    k-means centroid (not a k-medoids medoid, which is a different definition).
    """

    key: str
    label: str
    rgb: list[int]
    count: int
    offset: int
    node_name: str
    cohesion: Optional[float] = None
    centre_file_id: Optional[str] = None


class ShapeMapWardCut(BaseModel):
    """A height where the tree's own merge cost jumps -- a natural place to cut.

    Not a recommendation and not an optimum: it reports only that merging the
    next pair costs disproportionately more than the ones below it. Which cut to
    use is the reader's decision.
    """

    granularity: float
    height: float
    clusters: int


class ShapeMapWard(BaseModel):
    """The height Ward's tree was cut at, and the band that height came from.

    Only present when clustering='ward' AND granularity was given. The client
    needs the band to map a slider position onto the part of the tree that
    produces a usable number of clusters, and cut_height is the absolute
    dissimilarity that was actually applied -- the number that makes two cuts
    comparable, which a cluster count cannot do.
    """

    cut_height: float
    height_lo: float
    height_hi: float
    k_max: int
    granularity: float
    suggestions: list[ShapeMapWardCut] = []


class ShapeMapTreeNode(BaseModel):
    """One node of Ward's tree, folded to the levels the Granularity slider uses.

    A node is a cluster at one cut height; its children are what it splits into
    at the next level that actually splits it. Levels where a node does not
    split are skipped, which is what keeps a block of identical parts from
    becoming a long chain of one-point peelings.
    """

    id: int
    parent: int
    level: int
    count: int
    height: float
    cohesion: Optional[float] = None
    centre: Optional[str] = None
    filename: Optional[str] = None


class ShapeMapTreeResponse(BaseModel):
    """Ward's hierarchy for one population, small enough to send in one piece.

    ``file_ids`` and ``leaf`` are parallel: ``leaf[i]`` is the id of the DEEPEST
    node holding ``file_ids[i]``. A node's membership is the union of its
    descendants' leaves, which the client derives rather than the server
    repeating every level's membership. At most ``2 * k_max - 1`` nodes, so no
    paging is needed.
    """

    index: str
    count: int
    k_max: int
    levels: list[int]
    file_ids: list[str]
    leaf: list[int]
    nodes: list[ShapeMapTreeNode]


class CatalogueGroup(BaseModel):
    """One family: a Ward node that ended its branch, with every member.

    ``kind`` is the authoritative label and the one a client should present:

    * ``"confident"`` -- cohesion met the threshold over enough members
      (``stats.min_reached_parts``) for that to mean something.
    * ``"small"`` -- cohesion met the threshold over fewer members than that.
      Cohesion is entangled with size: the mean over a three-part group's
      three pairs clears 0.95 far more easily than the mean over a 180-part
      group's 16,110, so the same number is a much weaker claim here.
    * ``"mixed"`` -- cohesion never met it (including a group of one, which
      has no cohesion at all). The tree ran out; this is the remainder.

    ``reached`` is the older boolean, kept for compatibility. It is true for
    ``confident`` and ``small`` alike, which is exactly the distinction it
    cannot make -- read ``kind``.

    ``height`` is the node's Ward merge height: the absolute dissimilarity at
    which the family was joined. It does not vary with group size the way
    cohesion does, and all the nodes of one tree level were cut at the same
    height, so heights compare between groups where cohesions do not.

    ``file_ids`` is the whole membership, not a sample: the entry exists to be
    tagged and highlighted as a unit. ``centre_file_id`` is a full id so the
    client can fetch its thumbnail from
    ``GET /similarity/index/{name}/parts/{part_id}/thumbnail``; this endpoint
    returns no image of its own.

    ``tagged_parts`` is how many members already carry at least one tag (a
    part counts once however many it has), and ``tags`` names the tags found
    in the group, most frequent first and at most eight. Together they say
    whether this entry has been dealt with already, which is what a user
    working down a catalogue needs to see on the tile.
    """

    id: int
    count: int
    cohesion: Optional[float] = None
    height: Optional[float] = None
    level: int
    kind: str
    reached: bool
    tagged_parts: int = 0
    tags: list[str] = []
    centre_file_id: Optional[str] = None
    centre_filename: Optional[str] = None
    file_ids: list[str]


class CatalogueBucket(BaseModel):
    """"At least this many parts": groups and parts above one size mark."""

    min_parts: int
    groups: int
    parts: int


class CatalogueStats(BaseModel):
    """How good this cut of the tree was, for comparing one cut with another.

    Every field EXCEPT ``groups`` covers the whole frontier, before
    ``min_parts`` dropped anything -- the stats are about the threshold and the
    depth, and a figure that moved when a display filter changed would answer
    nothing. ``groups`` is the number of entries actually returned.

    The three classes partition the population: ``confident_parts +
    small_parts + mixed_parts`` is every part the catalogue was built over.
    ``buckets`` counts the confident groups only, since "how many families
    have at least five parts" should count neither the untight ones nor the
    ones too small to judge.

    These are the same figures the diagnostic tables are printed from, by the
    same functions, so the panel and an offline measurement cannot disagree.
    """

    groups: int
    min_reached_parts: int
    confident_groups: int
    confident_parts: int
    confident_largest: int
    confident_median: float
    confident_singletons: int
    small_groups: int
    small_parts: int
    small_largest: int
    small_median: float
    small_singletons: int
    mixed_groups: int
    mixed_parts: int
    mixed_largest: int
    mixed_median: float
    mixed_singletons: int
    buckets: list[CatalogueBucket]


class CatalogueResponse(BaseModel):
    """Every family in an index, cut from the Ward tree at one cohesion.

    The groups PARTITION the population: each part appears in exactly one
    ``file_ids``, and the lengths add up to ``total_parts`` less whatever
    ``min_parts`` left out. The part and group counts live in ``stats``, which
    describes the whole frontier regardless of ``min_parts``, so the filtering
    never hides how much of the index the catalogue actually accounts for.

    ``k_max`` is the depth the tree was actually built to, which can be lower
    than the one requested when the population cannot support it.
    """

    index: str
    threshold: float
    k_max: int
    total_parts: int
    omitted_by_min_parts: int
    stats: CatalogueStats
    groups: list[CatalogueGroup]


class ShapeMapRequest(BaseModel):
    """Optional body for POST /map: the drill-down subset, if any.

    The body exists only for ``only_file_ids``, which can hold thousands of
    ids and so cannot be a query parameter. Every other knob stays a query
    parameter, and a request with no body behaves exactly as it did before
    drill-down existed.
    """

    only_file_ids: Optional[list[str]] = None


class ShapeMapResponse(BaseModel):
    """A 2D/3D scatter layout of an index's parts, coloured by cluster or tag.

    ``points`` carries one entry per file id: normalized coordinates in the
    ``[-1, 1]`` cube and a ``cluster`` id, ORDERED so each category occupies a
    contiguous run described by ``groups``. The layout is deterministic for a
    given index and parameters (FAISS k-means seed is fixed). ``legend``
    summarises the mapped subset (by cluster, or by tag when colouring by tag).

    ``subset_id`` is set when the request carried ``only_file_ids``; pass it to
    ``/map/scs`` as ``subset`` to get the point cloud for the same points.

    ``outlier_threshold_effective`` is the outlier threshold that was actually
    applied. It is lower than the requested one when the request exceeded the
    highest threshold this clustering can support, which is the largest value
    that still leaves two points in their clusters; the request is capped there
    rather than abandoned, so raising the setting separates steadily more points
    and then stops changing.
    """

    index: str
    count: int
    clusters: int
    dims: int
    points: list[ShapeMapPoint]
    groups: list[ShapeMapGroup] = []
    legend: list[ShapeMapLegendEntry]
    subset_id: Optional[str] = None
    ward: Optional[ShapeMapWard] = None
    outlier_threshold_effective: Optional[float] = None


class ShapeMapLegendOrderResponse(BaseModel):
    """The order to DRAW the legend's cluster rows in, as cluster ids.

    Not a reordering of anything the map returned. ``groups`` is positional --
    the viewer resolves a pick through ``groups[node].offset`` -- so the array
    keeps the order it was built in and the client reads its rows through this
    list instead.

    ``order`` holds every cluster id the map produced, once each, with ids
    whose centroids are close to one another next to one another. ``method``
    says which leaf ordering produced it: ``"ward+olo"``, or ``"ward"`` when
    the cluster count is past the point where optimal leaf ordering is
    affordable in a request.
    """

    index: str
    k: int
    method: str
    order: list[int]


class IndexDeleteResponse(BaseModel):
    name: str
    deleted: bool


class IndexStatsResponse(BaseModel):
    name: str
    files: int
    bodies: int
    assemblies: int
    single_part: int
    default_clusters: int
    dim: Optional[int] = None
    schema_version: int
    last_modified: Optional[str] = None


class IndexModelConsistencyResponse(BaseModel):
    name: str
    current_ckpt_sha256: str
    total_files: int
    matching_files: int
    mismatching_files: int
    unknown_files: int
    status: str


class IndexDuplicateNamesItem(BaseModel):
    file_id: str
    names: list[str]


class IndexDuplicateNamesResponse(BaseModel):
    name: str
    total: int
    extra_copies: int
    offset: int
    limit: int
    items: list[IndexDuplicateNamesItem]


class IndexPartItem(BaseModel):
    id: str
    filename: Optional[str] = None
    kind: str
    bodies: int
    thumbnail_url: Optional[str] = None
    scs_url: Optional[str] = None
    registered_at: Optional[str] = None
    tags: list[str] = []


class TagCount(BaseModel):
    tag: str
    count: int


class IndexTagsResponse(BaseModel):
    tags: list[TagCount]
    total: int


class PartTagsResponse(BaseModel):
    part_id: str
    tags: list[str]


class PartTagsUpdate(BaseModel):
    tags: list[str]


class BulkTagUpdate(BaseModel):
    part_ids: list[str]


class SkippedPart(BaseModel):
    part_id: str
    reason: str


class BulkTagResponse(BaseModel):
    tag: str
    updated: int
    skipped: list[SkippedPart] = []


class TagDeleteResponse(BaseModel):
    tag: str
    removed_from: int


class TagVocabularyResponse(BaseModel):
    """The names the application offers to choose from, NOT the tags in use.

    One list for every index: the wording of a tag is a vocabulary choice, not
    a property of an index. A name stays here after the last part carrying it
    is untagged, which is the whole point of the vocabulary -- the wording is
    kept so it can be picked again instead of retyped.
    """

    tags: list[str]
    total: int


class TagVocabularyUpdate(BaseModel):
    tags: list[str]


class TagVocabularyAddResponse(TagVocabularyResponse):
    """``added`` is what this request put in; ``rejected`` what would not fit.

    Names already present appear in neither list. ``rejected`` is non-empty
    only when the vocabulary is at its limit, and is reported rather than
    dropped so a client never believes it registered a name that it did not.
    """

    added: list[str] = []
    rejected: list[str] = []


class TagVocabularyDeleteResponse(BaseModel):
    """``removed`` is False when the name was not there, which is not an error."""

    tag: str
    removed: bool
    tags: list[str]
    total: int


class ContextKeySummary(BaseModel):
    key: str
    count: int
    type: str


class ContextSummaryResponse(BaseModel):
    keys: list[ContextKeySummary]
    parts_with_context: int
    units: dict = {}


class PartContextResponse(BaseModel):
    part_id: str
    context: dict


class PartContextUpdate(BaseModel):
    context: dict


class ContextPredictionEntry(BaseModel):
    value: Optional[object] = None
    confidence: Optional[float] = None
    evidence_count: Optional[int] = None
    status: Optional[str] = None
    # The SDK reports per-key coverage as a breakdown dict (observed_hits,
    # total_hits, observed_score_weight, missing_score_weight), not a scalar.
    coverage: Optional[object] = None
    alternatives: Optional[object] = None
    reasons: Optional[object] = None
    injected_context: Optional[object] = None


class ContextPredictHit(BaseModel):
    id: str
    score: float
    filename: Optional[str] = None
    thumbnail_url: Optional[str] = None
    values: dict


class ContextPredictRequest(BaseModel):
    file_id: Optional[str] = None
    keys: list[str]
    top_k: int = 100
    kind: Literal["any", "part", "assembly"] = "part"
    tags: Optional[list[str]] = None
    tag_mode: Literal["any", "all"] = "any"
    tagged: Optional[bool] = None
    query_context: Optional[dict] = None
    status_policy: Optional[dict] = None
    # When present, the neighbours to estimate from are pinned to exactly these
    # ids (the web client's Results-tab shown tiles), instead of a fresh top_k
    # neighbourhood. top_k should match the caller's search so the search pool
    # still contains them; the tag filter and top_k cap are not re-applied.
    neighbor_ids: Optional[list[str]] = None


class ContextPredictResponse(BaseModel):
    keys: list[str]
    candidates_examined: int
    hits_considered: int
    hits_with_context: int
    coverage: float
    predictions: dict[str, ContextPredictionEntry]
    hits: list[ContextPredictHit]


class IndexPartsResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[IndexPartItem]


# ---------------------------------------------------------------------------
# POST /similarity/index/create
# ---------------------------------------------------------------------------
@router.post("/index/create", response_model=IndexCreateResponse)
def create_index(
    name: str = Query(..., description="Name of the new index (^[a-z0-9_-]{1,64}$)."),
):
    """Create a new empty named similarity index.

    * ``name`` must match ``^[a-z0-9_-]{1,64}$``.
    * The index uses the single embedding checkpoint the server resolves from
      ``HOOPS_AI_EMBEDDINGS_MODEL_NAME`` or the ``models/`` directories.
    * Returns **409** if an index with that name already exists.
    * Returns **422** if the name is invalid.
    """
    try:
        result = core.create_index(name)
        return IndexCreateResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create index: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /similarity/index/list
# ---------------------------------------------------------------------------


@router.get("/index/list", response_model=list[IndexInfo])
def list_indexes():
    """Return metadata for all known similarity indexes."""
    try:
        return [IndexInfo(**entry) for entry in index_registry.list_indexes()]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list indexes: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/index/add
# ---------------------------------------------------------------------------


@router.post("/index/add", response_model=IndexAddResponse)
def add_to_index(
    name: str = Query(..., description="Target index name."),
    file_ids: Optional[str] = Query(
        None,
        description="Comma-separated file_ids of already-uploaded CAD files.",
    ),
    files: Optional[List[UploadFile]] = File(None),
    zip_file: Optional[UploadFile] = File(None),
):
    """Register CAD parts in the named index.

    Input sources can be combined freely (same rules as ``POST /similarity/compare``):

    * ``file_ids`` -- comma-separated ``file_id`` values from previous uploads
    * ``files`` -- one or more multipart CAD file uploads
    * ``zip_file`` -- a single ZIP archive (auto-extracted, Zip Slip protected)

    Re-registering an existing part ID overwrites the old entry (no duplicates).
    ``added`` counts new parts; ``updated`` counts overwritten parts.

    The embeddings model is always the one recorded in the index at creation time
    (stored in ``model.json``).  Use ``PUT /similarity/settings`` to change the model
    before creating a new index.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    errors: list[dict] = []
    resolved: list[tuple[str, str]] = []

    if file_ids:
        for fid in [f.strip() for f in file_ids.split(",") if f.strip()]:
            try:
                path = storage.find_persistent_CAD_file(fid)
                parts = path.name.split("_", 1)
                display = parts[1] if len(parts) == 2 and len(parts[0]) == 64 else path.name
                resolved.append((fid, display))
            except RuntimeError as exc:
                errors.append({"file_id": fid, "detail": str(exc)})

    if files:
        for upload in files:
            if not (upload and upload.filename):
                continue
            try:
                fid, _, _ = storage.upload_CAD_file_persistent(upload)
                resolved.append((fid, upload.filename))
            except Exception as exc:
                errors.append({"filename": upload.filename or "unknown", "detail": str(exc)})

    if zip_file and zip_file.filename:
        try:
            zip_data = zip_file.file.read()
            _process_zip(zip_data, resolved, errors)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to process ZIP archive: {exc}"
            ) from exc

    if not resolved and not errors:
        raise HTTPException(
            status_code=422,
            detail="No valid input provided. Supply file_ids, files, or zip_file.",
        )

    all_file_ids = [fid for fid, _ in resolved]

    try:
        result = index_registry.add_to_index(name, all_file_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except core.SchemaVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to add to index: {exc}") from exc

    # merge router-level resolve errors into core-level errors
    result["errors"] = errors + result.get("errors", [])
    return IndexAddResponse(**result)


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/search
# ---------------------------------------------------------------------------


@router.post("/index/{name}/search", response_model=IndexSearchResponse)
def search_named_index(
    name: str,
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id from a previous upload."),
    top_k: int = Query(10, ge=1, description="Number of similar shapes to return."),
    kind: Literal["any", "part", "assembly"] = Query(
        "any",
        description=(
            "Restrict hits by part kind. 'any' (default) applies no filter and "
            "preserves the previous behaviour; use 'part' to match "
            "hoops_ai_native_bridge."
        ),
    ),
    include_self: bool = Query(
        False,
        description=(
            "When true and the query file is registered in this index, put it "
            "first with score 1.0. The total still respects top_k."
        ),
    ),
    tags: Optional[str] = Query(
        None,
        description=(
            "Comma-separated tags. Hits are filtered after ranking, so scores "
            "and order are unchanged; the count can fall below top_k."
        ),
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged hits, false = only untagged."
    ),
):
    """Search a named index for the most similar parts to a query shape.

    Supply **either** a file upload *or* a ``file_id``.  Returns an empty ``hits``
    list when the index contains zero entries (no error).  ``image_url`` is always
    ``null``; clients render their own gallery from the returned hits.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    try:
        if file_id:
            resolved_id = file_id
        elif file:
            resolved_id, _, _ = storage.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")

        result = search.search_index(
            name,
            resolved_id,
            top_k,
            kind=kind,
            include_self=include_self,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
        )

        return IndexSearchResponse(
            hits=[IndexSearchHit(**h) for h in result["hits"]],
            count=result["count"],
            image_url=None,
            tag_filter=result.get("tag_filter"),
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/warm
# ---------------------------------------------------------------------------


@router.post("/index/{name}/warm", response_model=IndexWarmResponse)
def warm_named_index(
    name: str,
    file_id: str = Query(..., description="file_id of the query to pre-warm."),
    top_k: int = Query(10, ge=1, description="Same top_k the real search will use."),
    kind: Literal["any", "part", "assembly"] = Query("any"),
    include_self: bool = Query(False),
    tags: Optional[str] = Query(None, description="Same tag filter the real search will use."),
    tag_mode: Literal["any", "all"] = Query("any"),
    tagged: Optional[bool] = Query(None),
):
    """Pre-run a search so the first real request is served from the cache.

    Runs the *exact* ``search_index`` path a real request would, with the same
    parameters, and discards the hits -- only the response cache is populated,
    so what a subsequent real search reads back is byte-for-byte what it would
    have computed itself. For a live demo, warm the queries you will show right
    after registration finishes, so none of them pays the ~20 s cold encode.

    Populating the cache requires it to be enabled; ``cache_enabled`` reports
    whether it is, so a warm against a server started with
    ``HOOPS_AI_SEARCH_CACHE=0`` is visibly a no-op rather than a silent one.

    The parameters must match the real search exactly, because ``top_k``,
    ``kind``, ``include_self`` and the tag filter are all part of the cache key:
    a warm on a different key just wastes the encode and is silently never hit.
    The defaults here (``top_k=10``, ``kind="any"``, ``include_self=False``) are
    *not* what the web client sends. The client always sends ``kind="part"``
    (``searchParts`` in ``web/src/api.js`` pins it), ``include_self=true``
    (``web/src/search.js``), and a fixed ``top_k`` of 301 (the ``SEARCH_TOP_K``
    constant in ``web/src/config.js``; there is no Top K field in the UI). To
    warm what the client will actually request, pass
    ``top_k=301&kind=part&include_self=true`` (and any tag filter it will use).
    ``top_k`` goes into the key as sent even if the index holds fewer parts, so
    a mismatched top_k produces a different, never-hit key.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    try:
        result = search.search_index(
            name,
            file_id,
            top_k,
            kind=kind,
            include_self=include_self,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
        )
        return IndexWarmResponse(
            status="warmed",
            file_id=file_id,
            count=result["count"],
            cache_enabled=search._search_cache_enabled(),
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Warm failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/search-assembly
# ---------------------------------------------------------------------------


@router.post("/index/{name}/search-assembly", response_model=AssemblySearchResponse)
def search_named_index_assembly(
    name: str,
    request: Request,
    file: Optional[UploadFile] = File(None),
    file_id: Optional[str] = Query(None, description="file_id from a previous upload."),
    top_k: int = Query(10, ge=1, description="Number of similar assemblies to return."),
    candidate_k: int = Query(
        30, ge=1, le=1000,
        description="Per-body shortlist size used to gather candidate assemblies.",
    ),
    sim_thresh: float = Query(
        0.80, ge=0.0, le=1.0,
        description="Minimum body-to-body cosine similarity for a matched pair.",
    ),
    bop_weight: float = Query(
        0.30, ge=0.0, le=1.0,
        description=(
            "Blend weight of the bag-of-parts composition score against the "
            "geometric score. 0 = geometry only."
        ),
    ),
    coverage_mode: Literal["symmetric", "containment", "jaccard"] = Query(
        "symmetric",
        description=(
            "Coverage denominator: 'symmetric' (larger side), 'containment' "
            "(query side, finds assemblies containing the query) or 'jaccard'."
        ),
    ),
    use_idf: bool = Query(
        True,
        description="Weight parts by cluster rarity (IDF) so common fasteners count less.",
    ),
    include_self: bool = Query(
        False,
        description=(
            "When true and the query file is registered in this index, put it "
            "first with score 1.0. The total still respects top_k."
        ),
    ),
    tags: Optional[str] = Query(
        None,
        description=(
            "Comma-separated tags. Hits are filtered after ranking, so scores "
            "and order are unchanged; the count can fall below top_k."
        ),
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged hits, false = only untagged."
    ),
):
    """Search a named index for assemblies similar to a query assembly.

    Supply **either** a file upload *or* a ``file_id``.  Unlike
    ``/index/{name}/search`` (which ranks individual parts), this matches whole
    assemblies: query bodies are paired one-to-one with candidate bodies by
    Hungarian assignment, and the result is blended with a bag-of-parts
    composition score.  Same algorithm and defaults as hoops_ai_native_bridge.

    Each hit reports the diagnostics behind its score: ``geom_score``,
    ``coverage``, ``matched_parts`` (matched pairs), ``candidate_parts`` (bodies
    in the candidate) and ``query_parts`` (bodies in the query).

    Returns an empty ``hits`` list when the index contains zero entries.
    """
    _validate_name_or_raise(name)

    try:
        if file_id:
            resolved_id = file_id
        elif file:
            resolved_id, _, _ = storage.upload_CAD_file_persistent(file)
        else:
            raise HTTPException(status_code=422, detail="Either 'file' or 'file_id' is required.")

        result = assembly.search_assembly_index(
            name,
            resolved_id,
            top_k,
            candidate_k=candidate_k,
            sim_thresh=sim_thresh,
            bop_weight=bop_weight,
            coverage_mode=coverage_mode,
            use_idf=use_idf,
            include_self=include_self,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
        )

        return AssemblySearchResponse(
            hits=[AssemblySearchHit(**h) for h in result["hits"]],
            count=result["count"],
            image_url=None,
            tag_filter=result.get("tag_filter"),
        )
    except HTTPException:
        raise
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except core.SchemaVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assembly search failed: {exc}") from exc


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/map
# ---------------------------------------------------------------------------


@router.post("/index/{name}/map", response_model=ShapeMapResponse)
def index_shape_map(
    name: str,
    n_clusters: int = Query(
        0,
        ge=-1,
        le=1024,
        description=(
            "Number of clusters for the k-means colouring. 0 picks "
            "min(max(2, N // 50), 24) automatically (a point-count heuristic); "
            "-1 selects the count in [2, min(24, N-1)] at the elbow of the "
            "inertia curve (data-driven, heavier). Changing this "
            "reuses the cached projection, so it is cheap for a fixed count."
        ),
    ),
    dims: int = Query(
        3,
        ge=2,
        le=3,
        description="Projection dimensionality: 2 or 3. The unused axis is 0.",
    ),
    tags: Optional[str] = Query(
        None,
        description=(
            "Comma-separated tags. When given, only matching parts are mapped "
            "and the projection + clustering are recomputed on that subset."
        ),
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged parts, false = only untagged."
    ),
    include_tags: bool = Query(
        False,
        description=(
            "Include each part's tags on every point. Off by default to keep the "
            "response small; the legend already summarises tags for the subset."
        ),
    ),
    color_by: Literal["cluster", "tag"] = Query(
        "cluster",
        description=(
            "Which colour scheme the legend describes: by k-means cluster or by "
            "tag. Same vocabulary as /map/scs, so the legend matches the colours "
            "baked into the point cloud for the same color_by."
        ),
    ),
    kind: Literal["any", "part", "assembly"] = Query(
        "any",
        description=(
            "Restrict the map to single parts or assemblies (>= 2 body rows). "
            "Like a tag filter, this re-projects and re-clusters on the subset "
            "only, so a parts-only map is not skewed by assembly centroids. The "
            "web client defaults this to 'part'."
        ),
    ),
    outlier_threshold: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Split points whose cosine similarity to their own cluster centroid "
            "is below this value into a separate 'Outliers' group, instead of "
            "leaving them as nominal members of the nearest cluster (k-means "
            "assigns every point to some cluster regardless of distance). 0 "
            "disables the pass, which is the default."
        ),
    ),
    projection: str = Query(
        "pca",
        description=(
            "How the aggregate vectors are flattened to 2D/3D for display: "
            "'pca' (the default) or 'umap'. Only the placement of the "
            "points "
            "changes -- the clustering runs on the full-dimensional vectors "
            "either way, so every part keeps the same cluster. Returns 503 "
            "when 'umap' is asked for and umap-learn is not installed."
        ),
    ),
    clustering: str = Query(
        "kmeans",
        description=(
            "Clustering algorithm: 'kmeans' (the default), 'ward' or "
            "'average'. The last two are agglomerative, so a larger k is an "
            "exact refinement of a smaller one -- raising the count splits the "
            "families already on screen instead of redrawing them. They differ "
            "in the merge rule: Ward minimises the growth in within-cluster "
            "variance and so prefers compact clusters of similar size, while "
            "average (UPGMA) merges on the mean distance between members and "
            "tolerates elongated families. Returns 400 when either is asked "
            "for on a map with more points than it can be run on."
        ),
    ),
    granularity: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Only for clustering='ward': cut the tree at a HEIGHT rather than "
            "to a fixed count, 0.0 coarsest (one cluster) to 1.0 finest. A "
            "height says 'anything closer than this is one family', so uniform "
            "branches stay whole while varied ones split -- something n_clusters "
            "cannot express. When given, n_clusters is ignored and the height "
            "that was applied comes back on the payload as 'ward'. Ignored for "
            "k-means."
        ),
    ),
    ward_max: int = Query(
        256,
        ge=2,
        le=4096,
        description=(
            "Only for a tree clustering (ward or average): the ceiling the "
            "granularity scale is read against, so granularity=1.0 means this "
            "many clusters (or the population, when it is smaller). "
            "Meaningless for k-means. Must be "
            "the same value /map/tree is given as k_max: the tree's leaves and "            "the map's clusters are otherwise no longer one for one, and the "
            "tree's colours, checkboxes and tagging all rest on that match. "
            "The ceiling is 4096 on all three routes together: the UI sends "
            "one Max clusters value to /map, /map/scs and /map/tree, so a "
            "value any one of them refuses is a value the control cannot "
            "offer."
        ),
    ),
    body: Optional[ShapeMapRequest] = Body(None),
):
    """Project a named index's parts into a 2D/3D similarity map.

    The stored per-body embeddings are read back from the index (no CAD file is
    re-embedded), aggregated to one point per file id, projected with PCA and
    coloured by FAISS k-means. Similar parts land close together; the layout is
    deterministic for a given index and parameters.

    An optional tag filter (``tags`` / ``tag_mode`` / ``tagged``, the same
    vocabulary as search) restricts the map to a subset, recomputing the layout
    on that subset only. The response carries a server-side ``legend`` (by tag,
    or by cluster when the subset is untagged).

    An optional body ``{"only_file_ids": [...]}`` drills into an explicit set of
    parts, re-projecting and re-clustering within it -- the axes are redrawn, so
    this is not a zoom. It composes with the tag and ``kind`` filters. The
    response then carries a ``subset_id`` to hand to ``/map/scs``.

    The first call for an index runs the projection (a few seconds on a large
    corpus) and caches it; subsequent calls -- including different ``n_clusters``
    or ``dims`` -- return quickly from the cache.

    Returns **404** when the index does not exist, and **422** when
    ``only_file_ids`` leaves fewer than two parts to project.
    """
    _validate_name_or_raise(name)
    try:
        result = shape_map.compute_index_shape_map(
            name,
            n_clusters=n_clusters,
            dims=dims,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
            include_tags=include_tags,
            color_by=color_by,
            kind=kind,
            only_file_ids=(body.only_file_ids if body else None),
            outlier_threshold=outlier_threshold,
            projection=projection,
            clustering=clustering,
            granularity=granularity,
            ward_max=ward_max,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except core.UmapUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except core.WardTooLarge as exc:
        # 400, not 503: nothing is missing or broken here. The request is one
        # this data cannot support, which the caller fixes by asking for a
        # different algorithm or a smaller subset.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shape map failed: {exc}") from exc

    # Hand the payload over whole. This used to be assembled field by field,
    # which meant a new top-level field added to the payload was silently
    # dropped -- no error, just null in the response. Nested fields (a group's
    # cohesion, a legend entry's centre_file_id) pass through untouched because
    # the sub-dicts go over in one piece, so only top-level additions break,
    # which is exactly the kind of gap that goes unnoticed. Pydantic coerces the
    # lists of dicts into their sub-models, and BaseModel ignores extra keys.
    return ShapeMapResponse(**result)


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/map/legend-order
# ---------------------------------------------------------------------------


@router.post(
    "/index/{name}/map/legend-order", response_model=ShapeMapLegendOrderResponse,
)
def index_shape_map_legend_order(
    name: str,
    n_clusters: int = Query(
        0, ge=-1, le=1024,
        description="The cluster count the map was built with. Same meaning as /map.",
    ),
    tags: Optional[str] = Query(
        None,
        description=(
            "Comma-separated tags. Must be the map's own filter: a different "
            "population is a different clustering, and so a different order."
        ),
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged parts, false = only untagged."
    ),
    kind: Literal["any", "part", "assembly"] = Query(
        "any", description="Restrict to single parts or assemblies, exactly as /map does."
    ),
    outlier_threshold: float = Query(
        0.0, ge=0.0, le=1.0,
        description=(
            "The map's outlier threshold. It decides which parts are in which "
            "cluster, so it moves the centroids and therefore this order."
        ),
    ),
    clustering: str = Query(
        "kmeans",
        description=(
            "Must be 'kmeans'. Ward's clusters are a cut of a tree that "
            "already says which families are near which, so there is no "
            "second order to compute for it; 'ward' returns 400."
        ),
    ),
    body: Optional[ShapeMapRequest] = Body(None),
):
    """Order the legend by similarity instead of by size.

    The legend lists clusters by member count descending, which says nothing
    about which clusters are alike: one family split across six clusters is six
    rows scattered down the list. This returns those same cluster ids in the
    leaf order of a Ward dendrogram over their centroids, so the rows of one
    family fall together.

    Take the arguments from the map that is on screen. The assignment is read
    back through ``/map``'s own entry point with them, so this orders the
    clusters the user is looking at and normally costs only the dendrogram.

    Nothing about the map changes: cluster ids, colours, the ``groups`` array
    and its offsets, and the baked ``.scs`` are all exactly what ``/map``
    returned. Only the order the client draws its rows in.

    Returns **404** when the index does not exist and **400** for
    ``clustering=ward``, an invalid tag filter, or a subset too small to map.
    """
    _validate_name_or_raise(name)
    try:
        result = shape_map.compute_index_legend_order(
            name,
            n_clusters=n_clusters,
            kind=kind,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
            only_file_ids=(body.only_file_ids if body else None),
            outlier_threshold=outlier_threshold,
            clustering=clustering,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except core.WardTooLarge as exc:
        # Same reading as /map: the data cannot support the request, which is
        # not a failure of anything here.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Legend order failed: {exc}"
        ) from exc

    return ShapeMapLegendOrderResponse(**result)


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/map/tree
# ---------------------------------------------------------------------------


@router.post("/index/{name}/map/tree", response_model=ShapeMapTreeResponse)
def index_shape_map_tree(
    name: str,
    tags: Optional[str] = Query(
        None,
        description=(
            "Comma-separated tags. When given, the tree is built over the "
            "matching parts only -- the same population /map would draw."
        ),
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged parts, false = only untagged."
    ),
    kind: Literal["any", "part", "assembly"] = Query(
        "any",
        description=(
            "Restrict the tree to single parts or assemblies (>= 2 body rows), "
            "exactly as /map does."
        ),
    ),
    k_max: int = Query(
        256,
        ge=2,
        le=4096,
        description=(
            "Target cluster count of the deepest level. The levels are 1, 2, 4, "
            "8, ... up to this, the same logarithmic scale the Granularity "
            "slider uses, so a slider position and a level are the same cut of "
            "the same tree. Bounds the response at 2 * k_max - 1 nodes. Must "
            "be the same value /map and /map/scs are given as ward_max: they "
            "read their granularity against that ceiling, so a different value "
            "here leaves the tree's leaves and the map's clusters describing "
            "different families. The ceiling is 4096 because /map/catalogue "
            "accepts that much and a catalogue MUST be built from the same "
            "k_max as the tree it is shown against -- a different k_max "
            "renumbers the nodes, and the ids a catalogue returns would "
            "resolve to nothing here. compute_index_shape_map_tree imposes no "
            "ceiling of its own, so this is a validation bound only."
        ),
    ),
    clustering: Literal["ward", "average"] = Query(
        "ward",
        description=(
            "Which linkage builds the tree -- the same value /map is given. "
            "There is no tree for 'kmeans', so it is not accepted here: a "
            "hierarchy is what this route returns. Must match the map on "
            "screen, or the node ids name families the map is not drawing."
        ),
    ),
    body: Optional[ShapeMapRequest] = Body(None),
):
    """The agglomerative hierarchy for an index's parts, folded to a handful of levels.

    No projection, colouring, cluster count or granularity is taken: a
    hierarchy has neither coordinates nor a legend, and the same tree answers
    every projection and colour mode, so those knobs would key nothing.

    An optional body ``{"only_file_ids": [...]}`` asks for the sub-tree of a
    drill-down. The subset must be a node of the tree, which a drill-down from
    a Ward map always is; anything else has no sub-tree and returns **400**.

    Returns **404** when the index does not exist, and **400** when the
    population is larger than Ward can be run on here or the subset is not a
    node.
    """
    _validate_name_or_raise(name)
    try:
        result = shape_map.compute_index_shape_map_tree(
            name,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
            kind=kind,
            only_file_ids=(body.only_file_ids if body else None),
            k_max=k_max,
            clustering=clustering,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except core.WardTooLarge as exc:
        # 400, not 503: nothing is missing or broken. The request is one this
        # data cannot support, which the caller fixes with a smaller subset.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Shape map tree failed: {exc}"
        ) from exc

    # Handed over whole, for the same reason /map does it: assembling the
    # response field by field is how a new top-level field goes missing.
    return ShapeMapTreeResponse(**result)


# ---------------------------------------------------------------------------
# POST /similarity/index/{name}/map/catalogue
# ---------------------------------------------------------------------------


@router.post("/index/{name}/map/catalogue", response_model=CatalogueResponse)
def index_catalogue(
    name: str,
    threshold: float = Query(
        0.95,
        gt=0,
        le=1,
        description=(
            "Minimum cohesion for a node to stand as a family. A branch is cut "
            "at the first node that reaches it; branches that never do are "
            "returned with reached=false."
        ),
    ),
    k_max: int = Query(
        2048,
        ge=2,
        le=4096,
        description=(
            "Depth of the tree the frontier is cut from. Measured on tmcad "
            "(7,928 parts): groups of five or more parts converge at 2048 and "
            "4096 adds none -- it adds only singletons, which reach any "
            "threshold trivially. Deeper is therefore not better here."
        ),
    ),
    kind: Literal["any", "part", "assembly"] = Query(
        "part",
        description=(
            "Restrict the catalogue to single parts or assemblies, as /map "
            "does. Defaults to parts: a catalogue is a list of things to make."
        ),
    ),
    min_parts: int = Query(
        1,
        ge=1,
        description=(
            "Drop groups smaller than this from 'groups'. The part totals "
            "above still cover the whole index, so what was dropped stays "
            "visible in omitted_by_min_parts."
        ),
    ),
    min_reached_parts: int = Query(
        5,
        ge=1,
        description=(
            "Size at which a group that met the threshold counts as "
            "'confident' rather than 'small'. A CLASSIFICATION line only: it "
            "never changes which nodes the frontier is made of, because the "
            "walk already stops at the first cohesive node and that node's "
            "children are smaller still. Five is where the catalogue stops "
            "moving with k_max on the measured index."
        ),
    ),
    tags: Optional[str] = Query(
        None,
        description=(
            "Comma-separated tags. When given, the catalogue is built over the "
            "matching parts only -- the same population /map/tree would use."
        ),
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged parts, false = only untagged."
    ),
    clustering: Literal["ward", "average"] = Query(
        "ward",
        description=(
            "Which linkage builds the tree the frontier is cut from -- the "
            "same value /map and /map/tree are given. The two linkages make "
            "different families, which is the point of being able to ask for "
            "either here."
        ),
    ),
):
    """Every family in an index, cut from the Ward tree at one cohesion.

    Read-only: it builds nothing and stores nothing beyond its own cache. The
    tree is the one ``/map/tree`` serves, so a group here is a node the tree
    panel can show, at the cohesion it displays.

    The groups partition the population -- each part in exactly one, the
    lengths adding up -- which is what lets a client tag a group as a unit
    without touching a part twice.

    No thumbnails are returned. Each group names its ``centre_file_id``; the
    client fetches the image from
    ``GET /similarity/index/{name}/parts/{part_id}/thumbnail``.

    Groups come back in the order the tree panel draws them -- a pre-order
    walk of the tree -- so the catalogue and the tree read down together. Sort
    client-side for any other order; the tree's cannot be recovered from a
    sorted list.

    Returns **404** when the index does not exist, and **400** when the
    population is larger than Ward can be run on here or the tag filter is
    invalid.
    """
    _validate_name_or_raise(name)
    try:
        result = shape_map.compute_index_catalogue(
            name,
            threshold=threshold,
            k_max=k_max,
            kind=kind,
            min_parts=min_parts,
            min_reached_parts=min_reached_parts,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
            clustering=clustering,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except core.WardTooLarge as exc:
        # 400 for the same reason /map/tree gives it: the data cannot support
        # the request, which is the caller's to narrow, not the server's fault.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Catalogue failed: {exc}"
        ) from exc

    return CatalogueResponse(**result)


# ---------------------------------------------------------------------------
# GET /similarity/index/{name}/map/scs
# ---------------------------------------------------------------------------


@router.get("/index/{name}/map/scs")
def index_shape_map_scs(
    name: str,
    n_clusters: int = Query(
        0,
        ge=-1,
        le=1024,
        description="Same as /map: 0 uses the point-count heuristic, -1 the optimal scan.",
    ),
    dims: int = Query(
        3, ge=2, le=3, description="Same as /map: projection dimensionality, 2 or 3."
    ),
    tags: Optional[str] = Query(
        None, description="Same as /map: comma-separated tag filter."
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="Same as /map: 'any' or 'all'."
    ),
    tagged: Optional[bool] = Query(
        None, description="Same as /map: true = tagged only, false = untagged only."
    ),
    color_by: Literal["cluster", "tag"] = Query(
        "cluster",
        description=(
            "Colour scheme baked into the point cloud: by k-means cluster or by "
            "each part's first tag. Baked in because a single-node point cloud "
            "cannot be recoloured per point in the viewer."
        ),
    ),
    kind: Literal["any", "part", "assembly"] = Query(
        "any", description="Same as /map: restrict to single parts or assemblies."
    ),
    subset: Optional[str] = Query(
        None,
        description=(
            "The subset_id returned by /map for a drill-down request. Selects "
            "the same parts, so the cloud matches that map point for point."
        ),
    ),
    outlier_threshold: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Same as /map. Must match the value /map was given, since it "
            "decides which points are baked into the 'Outliers' node."
        ),
    ),
    projection: str = Query(
        "pca",
        description=(
            "Same as /map: 'pca' (the default) or 'umap'. Must match "
            "the value "
            "/map was given, since it decides where every point is placed and "
            "the cloud is authored from that payload. Returns 503 when 'umap' "
            "is asked for and umap-learn is not installed."
        ),
    ),
    clustering: str = Query(
        "kmeans",
        description=(
            "Same as /map: 'kmeans' (the default) or 'ward'. Ward is "
            "hierarchical, so a larger k is an exact refinement of a smaller "
            "one. Must match the value /map was given, since it decides which "
            "baked node each point ends up in and the cloud is authored from "
            "that payload. Returns 400 when 'ward' is asked for on a map with "
            "more points than it can be run on."
        ),
    ),
    granularity: Optional[float] = Query(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Same as /map: only for clustering='ward', cut the tree at a height "
            "rather than to a fixed count, 0.0 coarsest to 1.0 finest. Must "
            "match the value /map was given, since it decides how many baked "
            "nodes there are and the cloud is authored from that payload. "
            "Ignored for k-means."
        ),
    ),
    ward_max: int = Query(
        256,
        ge=2,
        le=4096,
        description=(
            "Same as /map: only for a tree clustering (ward or average), the "
            "ceiling the granularity scale is read against, so "
            "granularity=1.0 means this "
            "many clusters (or the population, when it is smaller). Must match "
            "the value /map was given, since it decides how many baked nodes "
            "there are, and the value /map/tree is given as k_max, or the "
            "tree's leaves and the map's clusters are no longer one for one. "
            "Ignored for k-means. The ceiling is 4096 on all three routes "
            "together: the UI sends one Max clusters value to /map, /map/scs "
            "and /map/tree, so a value any one of them refuses is a value the "
            "control cannot offer."
        ),
    ),
):
    """Return the Shape Map as a HOOPS Web Viewer point-cloud ``.scs`` file.

    This is a GET (unlike POST ``/map``) because the HOOPS Web Viewer loads a
    point cloud directly from a URL. The ``.scs`` is authored from the same
    cached map payload ``/map`` returns, so point order -- and thus the file-id
    correspondence a viewer relies on -- is identical.

    Requires the ``scs_points`` build tool to be present in ``bin/``; when it is
    missing this route returns **503** while ``/map`` keeps working. Returns
    **404** for an unknown index, and **409** when ``subset`` is no longer held
    by the server (the subset registry is a small LRU) -- the client should
    re-post ``/map`` to obtain a fresh ``subset_id``.
    """
    _validate_name_or_raise(name)
    only_file_ids = None
    if subset:
        only_file_ids = shape_map._resolve_map_subset(subset)
        if only_file_ids is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Unknown or expired map subset '{subset}'. Re-post "
                    f"/similarity/index/{name}/map with the same only_file_ids "
                    "to obtain a fresh subset_id."
                ),
            )
    try:
        path = shape_map.compute_index_shape_map_scs(
            name,
            n_clusters=n_clusters,
            dims=dims,
            tags=_split_tags(tags),
            tag_mode=tag_mode,
            tagged=tagged,
            color_by=color_by,
            kind=kind,
            only_file_ids=only_file_ids,
            outlier_threshold=outlier_threshold,
            projection=projection,
            clustering=clustering,
            granularity=granularity,
            ward_max=ward_max,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except core.UmapUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except core.WardTooLarge as exc:
        # 400, not 503: nothing is missing or broken here. The request is one
        # this data cannot support, which the caller fixes by asking for a
        # different algorithm or a smaller subset.
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except core.ScsToolUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Shape map SCS failed: {exc}") from exc

    return FileResponse(str(path), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# GET /similarity/index/{name}/stats
# ---------------------------------------------------------------------------


@router.get("/index/{name}/stats", response_model=IndexStatsResponse)
def get_index_stats(name: str):
    """Return body-level statistics for a named index.

    ``files`` counts distinct CAD files, ``bodies`` counts vector rows,
    ``assemblies`` counts files with >= 2 bodies and ``single_part`` counts
    single-body files. Legacy (schema v1) indexes report one body per file.
    ``default_clusters`` is the cluster count the Shape Map should open with
    for this index, from the one rule startup warmup also builds with.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise
    try:
        stats = core.get_index_stats(name)
        # Counted over the single parts, not every file: the map opens on
        # Show: Parts (kind="part"), so those are the points that get
        # clustered. A count chosen against a larger population would ask for
        # more clusters than the map has points to fill.
        stats["default_clusters"] = shape_map.shape_map_auto_cluster_count(
            stats["single_part"]
        )
        return IndexStatsResponse(**stats)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /similarity/index/{name}/model-consistency
# ---------------------------------------------------------------------------


@router.get(
    "/index/{name}/model-consistency",
    response_model=IndexModelConsistencyResponse,
)
def get_index_model_consistency(name: str):
    """Report whether an index matches the embedding checkpoint in use now.

    Each record carries the SHA-256 of the checkpoint that produced it. This
    compares those against the checkpoint the server currently resolves and
    returns ``status`` = ``match`` / ``partial_mismatch`` / ``mismatch`` with the
    per-file counts.

    Hashing a multi-GB checkpoint takes a few seconds, so the **first** call
    after startup is slow; the digest is then cached for the process. This is a
    diagnostic endpoint: the check is never run on search or at startup, and its
    result is deliberately excluded from the index list and search responses.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise
    try:
        return IndexModelConsistencyResponse(
            **core.check_index_model_consistency(name)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to check model consistency: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# GET /similarity/index/{name}/duplicates
# ---------------------------------------------------------------------------


@router.get("/index/{name}/duplicates", response_model=IndexDuplicateNamesResponse)
def get_index_duplicate_names(
    name: str,
    offset: int = Query(0, ge=0, description="Number of items to skip."),
    limit: int = Query(500, ge=1, le=2000, description="Max items per page (1-2000)."),
):
    """List the geometries in an index that were registered under several names.

    A file_id is the SHA-256 of the file's bytes, so uploading the same
    geometry under different filenames produces **one** index entry, not
    several. Nothing here is duplicated in the index; what is duplicated is the
    original filename each copy arrived with, and only the CAD store remembers
    those. This endpoint is the one place that difference is visible.

    ``total`` counts the geometries carrying more than one original filename
    and ``extra_copies`` counts the redundant filenames (all of them minus one
    per geometry); both are for the whole index, not for the page. Only
    filenames are returned -- never a path on the server.

    Paginated like ``/parts`` even though the list is normally short: it is
    bounded by the index size, and an index built from a corpus with a large
    duplicated subset should not be able to turn one request into a response
    of unbounded size. The counts stay complete regardless of the page.

    A diagnostic, like ``/model-consistency``: it scans the CAD store once per
    call and is never run on search or at startup.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise
    try:
        result = core.get_index_duplicate_names(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list duplicate names: {exc}"
        ) from exc

    page = result["items"][offset : offset + limit]
    return IndexDuplicateNamesResponse(
        name=result["name"],
        total=result["total"],
        extra_copies=result["extra_copies"],
        offset=offset,
        limit=limit,
        items=[IndexDuplicateNamesItem(**it) for it in page],
    )


# ---------------------------------------------------------------------------
# GET /similarity/index/{name}/parts
# ---------------------------------------------------------------------------


@router.get("/index/{name}/parts", response_model=IndexPartsResponse)
def list_index_parts(
    name: str,
    request: Request,
    offset: int = Query(0, ge=0, description="Number of items to skip."),
    limit: int = Query(100, ge=1, le=2000, description="Max items per page (1-2000)."),
    kind: Optional[str] = Query(
        None, description="Filter by kind: 'part' or 'assembly'. Omit for all."
    ),
    tags: Optional[str] = Query(
        None, description="Comma-separated tags to filter by. Omit for no tag filter."
    ),
    tag_mode: Literal["any", "all"] = Query(
        "any", description="'any' matches parts carrying at least one tag, 'all' requires every tag."
    ),
    tagged: Optional[bool] = Query(
        None, description="true = only tagged parts, false = only untagged. Omit for both."
    ),
):
    """Return a paginated, file-level listing of a named index.

    ``thumbnail_url`` / ``scs_url`` are absolute URLs to the per-part asset
    endpoints, or ``null`` when the asset does not exist on disk. Tag filters
    are applied before pagination, so ``total`` reflects the filter.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise
    if kind is not None and kind not in ("part", "assembly"):
        raise HTTPException(
            status_code=422, detail="kind must be 'part' or 'assembly' when supplied."
        )
    tag_list = _split_tags(tags)
    try:
        result = core.list_index_parts(
            name,
            offset=offset,
            limit=limit,
            kind=kind,
            tags=tag_list,
            tag_mode=tag_mode,
            tagged=tagged,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list parts: {exc}") from exc

    base = str(request.base_url).rstrip("/")
    items = []
    for it in result["items"]:
        pid = it["id"]
        thumbnail_url = (
            f"{base}/similarity/index/{name}/parts/{pid}/thumbnail"
            if it.get("has_thumbnail")
            else None
        )
        scs_url = (
            f"{base}/similarity/index/{name}/parts/{pid}/scs"
            if it.get("has_scs")
            else None
        )
        items.append(
            IndexPartItem(
                id=pid,
                filename=it.get("filename"),
                kind=it["kind"],
                bodies=it["bodies"],
                thumbnail_url=thumbnail_url,
                scs_url=scs_url,
                registered_at=it.get("registered_at"),
                tags=it.get("tags") or [],
            )
        )
    return IndexPartsResponse(
        total=result["total"], offset=result["offset"], limit=result["limit"], items=items
    )


# ---------------------------------------------------------------------------
# GET /similarity/index/{name}/parts/{part_id}/thumbnail | /scs
# ---------------------------------------------------------------------------


@router.get("/index/{name}/parts/{part_id}/thumbnail")
def get_index_part_thumbnail(name: str, part_id: str):
    """Return the stored PNG thumbnail for a registered part (404 if missing)."""
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise
    try:
        path = index_registry._resolve_index_asset(name, part_id, "thumbnail")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    return FileResponse(str(path), media_type="image/png")


@router.get("/index/{name}/parts/{part_id}/scs")
def get_index_part_scs(name: str, part_id: str):
    """Return the stored SCS stream cache for a registered part (404 if missing)."""
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise
    try:
        path = index_registry._resolve_index_asset(name, part_id, "scs")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="SCS file not found.")
    return FileResponse(str(path), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Index tags
# ---------------------------------------------------------------------------
#
# Tags are a production feature of the shared server. Every response carries an
# ETag, and every mutation accepts If-Match so two people editing the same index
# cannot silently overwrite one another.


def _raise_tag_error(exc: Exception) -> None:
    """Map core tag errors onto HTTP status codes."""
    if isinstance(exc, core.TagsPreconditionFailed):
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, core.TagsFileError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=f"Tag operation failed: {exc}") from exc


@router.get("/index/{name}/tags", response_model=IndexTagsResponse)
def list_index_tags(name: str, response: Response):
    """List every tag used in the index with the number of parts carrying it.

    Ordered by descending count, then by tag name. The ``ETag`` header is the
    value to send back as ``If-Match`` when updating. This is the tags IN USE
    in this index; the names available to choose from are a separate,
    application-wide list, returned by ``GET /tag-vocabulary``.
    """
    _validate_name_or_raise(name)
    try:
        result, etag = core.get_index_tags(name)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    response.headers["ETag"] = etag
    return IndexTagsResponse(
        tags=[TagCount(**t) for t in result["tags"]], total=result["total"]
    )


@router.get("/index/{name}/parts/{part_id}/tags", response_model=PartTagsResponse)
def get_index_part_tags(name: str, part_id: str, response: Response):
    """Return the tags of one part (an empty list when it has none)."""
    _validate_name_or_raise(name)
    try:
        result, etag = core.get_part_tags(name, part_id)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    response.headers["ETag"] = etag
    return PartTagsResponse(**result)


@router.put("/index/{name}/parts/{part_id}/tags", response_model=PartTagsResponse)
def put_index_part_tags(
    name: str,
    part_id: str,
    response: Response,
    payload: PartTagsUpdate,
    if_match: Optional[str] = Header(
        None,
        alias="If-Match",
        description=(
            "ETag from a previous read. Returns 412 when the tags have changed "
            "since. Omitting it skips the check (last write wins)."
        ),
    ),
    client_id: Optional[str] = Header(
        None, alias="X-Client-Id", description="Recorded as updated_by."
    ),
):
    """Replace the entire tag set of a part. An empty list removes all tags."""
    _validate_name_or_raise(name)
    try:
        result, etag = core.set_part_tags(
            name, part_id, payload.tags, if_match=if_match, client_id=client_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    response.headers["ETag"] = etag
    return PartTagsResponse(**result)


@router.post("/index/{name}/tags/{tag}/parts", response_model=BulkTagResponse)
def assign_index_tag(
    name: str,
    tag: str,
    response: Response,
    payload: BulkTagUpdate,
    if_match: Optional[str] = Header(None, alias="If-Match"),
    client_id: Optional[str] = Header(None, alias="X-Client-Id"),
):
    """Add *tag* to several parts at once.

    Ids that are not registered in this index are reported in ``skipped``
    rather than failing the request, so one stale id cannot discard the rest.
    """
    _validate_name_or_raise(name)
    if not payload.part_ids:
        raise HTTPException(status_code=422, detail="No part_ids supplied.")
    try:
        result, etag = core.assign_tag(
            name, tag, payload.part_ids, if_match=if_match, client_id=client_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    response.headers["ETag"] = etag
    return BulkTagResponse(
        tag=result["tag"],
        updated=result["updated"],
        skipped=[SkippedPart(**s) for s in result["skipped"]],
    )


@router.delete("/index/{name}/tags/{tag}/parts", response_model=BulkTagResponse)
def unassign_index_tag(
    name: str,
    tag: str,
    response: Response,
    payload: BulkTagUpdate,
    if_match: Optional[str] = Header(None, alias="If-Match"),
    client_id: Optional[str] = Header(None, alias="X-Client-Id"),
):
    """Remove *tag* from several parts at once. Same skip policy as assignment."""
    _validate_name_or_raise(name)
    if not payload.part_ids:
        raise HTTPException(status_code=422, detail="No part_ids supplied.")
    try:
        result, etag = core.unassign_tag(
            name, tag, payload.part_ids, if_match=if_match, client_id=client_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    response.headers["ETag"] = etag
    return BulkTagResponse(
        tag=result["tag"],
        updated=result["updated"],
        skipped=[SkippedPart(**s) for s in result["skipped"]],
    )


@router.delete("/index/{name}/tags/{tag}", response_model=TagDeleteResponse)
def delete_index_tag(
    name: str,
    tag: str,
    response: Response,
    confirm: bool = Query(
        False, description="Must be ``true``: this removes the tag from every part."
    ),
    if_match: Optional[str] = Header(None, alias="If-Match"),
    client_id: Optional[str] = Header(None, alias="X-Client-Id"),
):
    """Delete a tag from the whole index.

    Requires ``?confirm=true`` (409 without it), following the same rule as
    ``DELETE /similarity/index/{name}``.
    """
    _validate_name_or_raise(name)
    if not confirm:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Removing tag '{tag}' from every part of index '{name}' cannot "
                "be undone. Re-send this request with ?confirm=true to proceed."
            ),
        )
    try:
        result, etag = core.delete_tag(
            name, tag, if_match=if_match, client_id=client_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    response.headers["ETag"] = etag
    return TagDeleteResponse(**result)


# ---------------------------------------------------------------------------
# Tag vocabulary -- DATA_ROOT/tag_vocabulary.json (one per application)
# ---------------------------------------------------------------------------
#
# The names in use, kept apart from which parts carry them, so that untagging
# everything does not lose the wording. One list for the whole application: a
# name is a vocabulary choice rather than a property of an index. A client
# offers the union of this list and the tags in use in the index it is looking
# at (GET /index/{name}/tags), in-use names first.
#
# These routes take no index name for that reason, and live here beside the
# per-index tag routes because they are the same subject to a client.


@router.get("/tag-vocabulary", response_model=TagVocabularyResponse)
def get_tag_vocabulary():
    """List the registered tag names, in the file's own order.

    No index name: there is one vocabulary for the whole application, because
    the wording of a tag does not belong to an index. A name is here because it
    was assigned to a part at some point -- in any index -- or because it was
    registered ahead of time. It stays after the last part carrying it is
    untagged. No ``ETag``: the list grows by name and the endpoints below are
    additive per name, so there is no lost-update to guard against.
    """
    try:
        result = core.get_tag_vocabulary()
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    return TagVocabularyResponse(**result)


@router.post("/tag-vocabulary", response_model=TagVocabularyAddResponse)
def add_tag_vocabulary(payload: TagVocabularyUpdate):
    """Register names ahead of use. Names already present are left as they are.

    This does not tag anything; it only makes the names available to pick, in
    every index.
    """
    try:
        result = core.add_tag_vocabulary(payload.tags)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    return TagVocabularyAddResponse(**result)


@router.delete("/tag-vocabulary/{tag}", response_model=TagVocabularyDeleteResponse)
def delete_tag_vocabulary(tag: str):
    """Take one name out of the vocabulary. Parts carrying it KEEP the tag.

    This is for tidying away a mistyped or abandoned name. It changes nothing
    about the parts, in any index: the tag goes on appearing in
    ``GET /index/{name}/tags`` and in every tag filter for as long as a part
    carries it. To remove a tag from the parts of one index, use
    ``DELETE /index/{name}/tags/{tag}``. No ``?confirm`` for that reason --
    nothing here can be lost but the suggestion itself.
    """
    try:
        result = core.remove_tag_vocabulary(tag)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_tag_error(exc)
    return TagVocabularyDeleteResponse(**result)


# ---------------------------------------------------------------------------
# Index context (production records) -- indexes/<name>/context.json
# ---------------------------------------------------------------------------
#
# Context records mirror tags: a per-index sidecar, an ETag on every read and
# If-Match on every write. The predict endpoint reads the records of a query
# part's nearest neighbours and returns a value for each requested field.


def _raise_context_error(exc: Exception) -> None:
    """Map core/context_store errors onto HTTP status codes."""
    if isinstance(exc, context_store.ContextPreconditionFailed):
        raise HTTPException(status_code=412, detail=str(exc)) from exc
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, PermissionError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, context_store.ContextFileError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise HTTPException(
        status_code=500, detail=f"Context operation failed: {exc}"
    ) from exc


@router.post("/index/{name}/predict-context", response_model=ContextPredictResponse)
def predict_index_context(name: str, payload: ContextPredictRequest, request: Request):
    """Predict production records for a query part from its nearest neighbours.

    The query part is given by ``file_id`` (a previously uploaded/registered
    part). It is searched exactly as ``/index/{name}/search`` searches --
    ``kind`` defaults to ``"part"`` so an estimate is not diluted by assemblies
    -- and the records of the resulting neighbours are aggregated into one value
    per requested key. ``hits`` returns the evidence records that fed the
    prediction (with ``thumbnail_url`` for display); ``coverage`` is the share
    of the considered neighbours that carried any record. ``status_policy``
    optionally tunes the ready/needs-review threshold.
    """
    _validate_name_or_raise(name)
    if not payload.file_id:
        raise HTTPException(status_code=422, detail="'file_id' is required.")
    try:
        result = core.predict_context(
            name,
            payload.file_id,
            payload.keys,
            top_k=payload.top_k,
            kind=payload.kind,
            tags=payload.tags,
            tag_mode=payload.tag_mode,
            tagged=payload.tagged,
            query_context=payload.query_context,
            status_policy=payload.status_policy,
            neighbor_ids=payload.neighbor_ids,
        )
    except HTTPException:
        raise
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        _raise_context_error(exc)
    base = str(request.base_url).rstrip("/")
    hits = [
        ContextPredictHit(
            id=h["id"],
            score=h["score"],
            filename=h.get("filename"),
            thumbnail_url=(
                f"{base}/similarity/index/{name}/parts/{h['id']}/thumbnail"
                if h.get("has_thumbnail")
                else None
            ),
            values=h.get("values", {}),
        )
        for h in result["hits"]
    ]
    return ContextPredictResponse(
        keys=result["keys"],
        candidates_examined=result["candidates_examined"],
        hits_considered=result["hits_considered"],
        hits_with_context=result["hits_with_context"],
        coverage=result["coverage"],
        predictions={
            k: ContextPredictionEntry(**v) for k, v in result["predictions"].items()
        },
        hits=hits,
    )


@router.get("/index/{name}/context", response_model=ContextSummaryResponse)
def get_index_context_summary(name: str, response: Response):
    """Summarise the context fields present in the index and how many parts use each.

    The ``ETag`` header is the value to send back as ``If-Match`` when writing a
    part's context.
    """
    _validate_name_or_raise(name)
    try:
        result, etag = core.get_index_context_summary(name)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_context_error(exc)
    response.headers["ETag"] = etag
    return ContextSummaryResponse(
        keys=[ContextKeySummary(**k) for k in result["keys"]],
        parts_with_context=result["parts_with_context"],
        units=result.get("units", {}),
    )


@router.get(
    "/index/{name}/parts/{part_id}/context", response_model=PartContextResponse
)
def get_index_part_context(name: str, part_id: str, response: Response):
    """Return the stored context of one part (an empty object when it has none)."""
    _validate_name_or_raise(name)
    try:
        result, etag = core.get_part_context(name, part_id)
    except HTTPException:
        raise
    except Exception as exc:
        _raise_context_error(exc)
    response.headers["ETag"] = etag
    return PartContextResponse(**result)


@router.put(
    "/index/{name}/parts/{part_id}/context", response_model=PartContextResponse
)
def put_index_part_context(
    name: str,
    part_id: str,
    response: Response,
    payload: PartContextUpdate,
    if_match: Optional[str] = Header(
        None,
        alias="If-Match",
        description=(
            "ETag from a previous read. Returns 412 when the context has changed "
            "since. Omitting it skips the check (last write wins)."
        ),
    ),
    client_id: Optional[str] = Header(
        None, alias="X-Client-Id", description="Recorded as updated_by."
    ),
):
    """Replace the entire context of a part. An empty object removes it."""
    _validate_name_or_raise(name)
    try:
        result, etag = core.set_part_context(
            name, part_id, payload.context, if_match=if_match, client_id=client_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        _raise_context_error(exc)
    response.headers["ETag"] = etag
    return PartContextResponse(**result)


# ---------------------------------------------------------------------------
# DELETE /similarity/index/{name}/parts
# ---------------------------------------------------------------------------


@router.delete("/index/{name}/parts", response_model=IndexRemoveResponse)
def remove_parts_from_index(
    name: str,
    part_ids: str = Query(..., description="Comma-separated part IDs (file_ids) to remove."),
):
    """Remove registered parts from a named index by their part IDs (file_ids)."""
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    ids = [p.strip() for p in part_ids.split(",") if p.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="No part_ids supplied.")

    try:
        result = core.remove_from_index(name, ids)
        return IndexRemoveResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except core.SchemaVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (core.EnvConfigError, core.PathConfigError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Remove failed: {exc}") from exc


# ---------------------------------------------------------------------------
# DELETE /similarity/index/{name}
# ---------------------------------------------------------------------------


@router.delete("/index/{name}", response_model=IndexDeleteResponse)
def delete_index(
    name: str,
    confirm: bool = Query(
        False,
        description="Must be ``true`` to confirm destructive deletion.",
    ),
):
    """Delete a named index and all its on-disk data.

    * Requires ``?confirm=true``; without it returns **409** with an instruction.
    * Returns **404** when the index does not exist.
    """
    try:
        _validate_name_or_raise(name)
    except HTTPException:
        raise

    if not confirm:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Deletion of index '{name}' is destructive and irreversible. "
                "Re-send this request with ?confirm=true to proceed."
            ),
        )

    try:
        result = core.delete_index(name)
        return IndexDeleteResponse(**result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helper -- validate index name at router boundary
# ---------------------------------------------------------------------------


def _split_tags(tags: Optional[str]) -> Optional[list[str]]:
    """Split a comma-separated tag query parameter, or return None if absent.

    An empty or whitespace-only value means "no tag filter" rather than "match
    the empty tag", so a client can send tags= unconditionally.
    """
    if tags is None:
        return None
    parts = [t.strip() for t in tags.split(",")]
    parts = [t for t in parts if t]
    return parts or None


def _validate_name_or_raise(name: str) -> None:
    """Convert core validation errors to HTTPExceptions at the router boundary."""
    try:
        core._validate_index_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ===========================================================================
# ZIP extraction helper (shared by /compare and /index/add)
# ===========================================================================


def _process_zip(
    zip_data: bytes,
    resolved: list[tuple[str, str]],
    errors: list[dict],
) -> None:
    """Extract a ZIP archive and upload each valid CAD file persistently.

    Raises ``HTTPException(400)`` on Zip Slip and ``HTTPException(413)`` when
    the uncompressed size or file count exceeds the configured limits.
    """
    try:
        new_resolved, new_errors = core.extract_zip_cad_files(
            zip_data, max_total_bytes=_ZIP_MAX_TOTAL_BYTES, max_files=_ZIP_MAX_FILES
        )
    except core.ZipSlipError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (core.ZipSizeLimitError, core.ZipFileLimitError) as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    resolved.extend(new_resolved)
    errors.extend(new_errors)

