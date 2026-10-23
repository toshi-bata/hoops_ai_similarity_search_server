"""The one read-only endpoint the web client uses to learn about the server.

The client is a static bundle, so everything else it knows is baked in at build
time (see web/src/config.js). This endpoint is the exception: a handful of
settings that only the server can answer, fetched once at startup.

**Only add a field here that the client must act on, and that leaks nothing.**
The license key, the SDK directory, the data root and every other path stay
server-side: they would tell a browser (and anyone watching it) about the host's
filesystem and entitlements for no benefit. Everything served here is already
observable from the UI's behaviour.
"""

from typing import Optional

import core
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(tags=["Client"])


class ClientConfigResponse(BaseModel):
    """Server settings the web client needs. One flat object, so a later
    setting is one more field and no client change beyond reading it."""

    demo: bool = Field(
        default=False,
        description=(
            "Whether the webinar-demo-only UI (the Estimate panel) is shown. "
            "From HOOPS_AI_DEMO; off unless a deployment asks for it."
        ),
    )

    umap_available: bool = Field(
        default=False,
        description=(
            "Whether the UMAP map projection can be chosen. umap-learn is an "
            "optional dependency, so a deployment without it must not offer the "
            "option at all rather than let the user pick one that returns 503. "
            "A yes/no about an installed Python package: it names no path and "
            "carries no entitlement information."
        ),
    )

    warm_clusterings: list[str] = Field(
        default_factory=list,
        description=(
            "Which clusterings startup warmup actually made hot, in the order "
            "HOOPS_AI_WARMUP_CLUSTERING names them. Empty means nothing is "
            "warm: HOOPS_AI_WARMUP_INDEXES is empty, or HOOPS_AI_WARMUP_STAGES "
            "does not include 'map'. The client uses its remembered choice "
            "when that choice is in this list, and falls back to a warm one "
            "when it is not -- opening the map on a clustering that was not "
            "warmed means the first user pays the full build, which is exactly "
            "what warmup is there to avoid. This is a UI default and nothing "
            "more -- it names no path and carries no entitlement information, "
            "as required by this module's header."
        ),
    )

    default_clustering: Optional[str] = Field(
        default=None,
        description=(
            "The first entry of warm_clusterings, or null when nothing is "
            "warm. Kept for clients that read this field before "
            "warm_clusterings existed; prefer that list, which lets a client "
            "keep a remembered choice that is warm instead of being pushed "
            "onto the one a deployment happens to lead with. A UI default and "
            "nothing more -- it names no path and carries no entitlement "
            "information, as required by this module's header."
        ),
    )

    default_ward_max: Optional[int] = Field(
        default=None,
        description=(
            "The Max clusters the Shape Map opens with under Ward: the finest "
            "cluster count its tree descends to. HOOPS_AI_WARMUP_WARD_MAX, or "
            "null when it is unset. Null means the client should use the "
            "per-index default_clusters from /index/{name}/stats, which is the "
            "same rule startup warmup builds with; a value means the operator "
            "has fixed one ceiling for every index. Either way the client must "
            "open on what was warmed: a map cut to a different ceiling is a "
            "different cache key, so a mismatch makes the first user pay the "
            "full build. A UI default and nothing more -- it names no path and "
            "carries no entitlement information, as required by this module's "
            "header."
        ),
    )


@router.get("/client-config", response_model=ClientConfigResponse)
def get_client_config():
    """Return the server settings the web client needs.

    Cheap and side-effect free: the client fetches it once when it starts.
    """
    warm = _warm_clusterings()
    return ClientConfigResponse(
        demo=core.demo_enabled(),
        umap_available=core._shape_map_umap_available(),
        warm_clusterings=warm,
        default_clustering=warm[0] if warm else None,
        default_ward_max=core.warmup_ward_max_setting(),
    )


def _warm_clusterings() -> list[str]:
    """The clusterings startup warmup makes hot, or [] if nothing is warmed.

    Whether HOOPS_AI_WARMUP_CLUSTERING is set is deliberately not asked: when
    it is unset, warmup still runs, on the default that
    warmup_clustering_setting() returns (k-means). That default is what is
    actually hot, so it is what the client has to be told about. Reading the
    raw variable instead once made the server answer "nothing is warm" for the
    commonest deployment of all, which let a browser that remembered Ward open
    the map on a cold build.
    """
    if not core.warmup_indexes_setting():
        return []
    if "map" not in core.warmup_stages_setting():
        return []
    return list(core.warmup_clustering_setting())
