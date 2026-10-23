#!/usr/bin/env python
"""Seed an index's context.json with synthetic production records (sample data).

This is a **sample** tool for the Context Layer demo. It fabricates production
records -- material, process, cost, lead time and so on -- so the
``/predict-context`` endpoint has something to aggregate. The values are
invented; they are not real manufacturing data.

It runs in two stages so the slow, deterministic part (hashing a CAD tree) is
separate from writing the sidecar:

Stage A -- ``scan``
    Walk a source tree, SHA-256 every file, and write a CSV of
    ``file_id,family,source_path`` sorted by id. ``family`` is the first path
    component under the source root, **derived from the file content hash, not
    the file name**: the same content stored under two families is one id with
    two rows, and that cross-family count is reported.

Stage B -- ``seed``
    Read that CSV and the ids registered in a target index, then write
    ``indexes/<name>/context.json``. Each record is synthesized deterministically
    from ``sha256(file_id + ":" + key)`` -- not ``random`` and not dict order --
    so re-running produces a byte-identical file. Roughly 50% of parts get a
    complete record, 30% get string fields only, and 20% get none.

Usage
-----
    python tools/seed_context_store.py scan --source <cad_root> --out families.csv
    python tools/seed_context_store.py seed --families families.csv --index <name> [--dry-run] [--force]

Run ``seed`` with the server stopped: it writes context.json directly rather
than through the API, so a concurrent edit made through the API could be lost.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import sys
from typing import Iterable, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_HASH_CHUNK = 1024 * 1024

# The stored marker on every synthesized record, so no one mistakes this sample
# data for real production data.
SYNTHETIC_SOURCE = "synthetic sample data"

# Value pools are family-dependent: the material, primary process, heat
# treatment and process route a part carries are drawn from that part family's
# own palette, not from one global pool. This is what makes the demo work --
# because neighbours of a shape query belong to the same family, they agree on
# these values, and the predictor has real evidence to aggregate. A single
# global pool would make every value independent and identically distributed,
# so no prediction could ever be justified. The family palettes below are
# modelled on the tutorial's synthetic PLM (embeddings_pipeline/database.py):
# each candidate is something a process engineer for that family would actually
# specify.

# Relative material cost multiplier. Deliberately a narrow band around 1.0:
# family-to-family cost separation is carried by ``base_cost`` (which spans an
# order of magnitude), so the material only nudges a part's cost within its
# family. A wide band here would make one family's cost spread swamp the
# family-typical value and turn "similar parts, similar cost" into noise.
_MATERIAL_MULTIPLIER: dict[str, float] = {
    "Mild Steel (S235JR)": 0.85,
    "Carbon Steel (C35)": 0.88,
    "Carbon Steel (C45)": 0.90,
    "Alloy Steel (42CrMo4)": 1.00,
    "Case-Hardening Steel (16MnCr5)": 1.02,
    "Case-Hardening Steel (20MnCr5)": 1.05,
    "Case-Hardening Steel (18CrNiMo7-6)": 1.12,
    "Bearing Steel (100Cr6)": 1.05,
    "Chrome Steel (AISI 52100)": 1.03,
    "Stainless Steel (AISI 304)": 1.15,
    "Stainless Steel (AISI 316)": 1.20,
    "Stainless Steel (AISI 440C)": 1.18,
    "Stainless Steel (A2-70)": 1.15,
    "Aluminum (6061-T6)": 1.00,
    "Gray Cast Iron (EN-GJL-250)": 0.82,
    "Ductile Iron (EN-GJS-500)": 0.86,
    "Property Class 8.8 Steel": 0.90,
    "Property Class 10.9 Steel": 0.98,
    "Property Class 12.9 Steel": 1.05,
}

# Relative process multiplier (labor + machine complexity), also a narrow band.
_PROCESS_MULTIPLIER: dict[str, float] = {
    "Cold Heading": 0.85,
    "Cold Forming": 0.85,
    "Casting": 0.85,
    "Thread Rolling": 0.90,
    "Drilling & Tapping": 0.90,
    "Sheet-Metal Forming": 0.92,
    "Forging": 0.95,
    "Turning": 0.98,
    "CNC Milling": 1.00,
    "Cylindrical Grinding": 1.08,
    "Hard Turning": 1.10,
    "Broaching": 1.10,
    "Hobbing": 1.15,
    "Gear Shaping": 1.20,
    "Superfinishing": 1.25,
    "Gear Grinding": 1.30,
}

# Per-family manufacturing profile. ``materials`` / ``processes`` /
# ``heat_treatments`` are weighted so a dominant choice emerges (consensus)
# while realistic alternatives still appear. ``route`` is the single process
# flow an expert would name for that family. ``base_cost`` / ``base_lead`` set
# the family's cost and lead-time scale so different families separate cleanly.
_FAMILY_PROFILES: dict[str, dict] = {
    "bearing": {
        "materials": [
            ("Bearing Steel (100Cr6)", 62),
            ("Chrome Steel (AISI 52100)", 24),
            ("Stainless Steel (AISI 440C)", 14),
        ],
        "processes": [
            ("Hard Turning", 45), ("Cylindrical Grinding", 35), ("Superfinishing", 20),
        ],
        "heat_treatments": [("Through-Hardening", 65), ("Case Carburizing", 35)],
        "route": (
            "Forged/bar blank -> Turning of races -> Through-hardening -> "
            "Raceway grinding -> Superfinishing -> Assembly of rolling elements -> Inspection"
        ),
        "base_cost": 90.0, "base_lead": 30,
    },
    "bolt": {
        "materials": [
            ("Property Class 8.8 Steel", 60),
            ("Property Class 10.9 Steel", 28),
            ("Stainless Steel (A2-70)", 12),
        ],
        "processes": [("Cold Heading", 55), ("Thread Rolling", 45)],
        "heat_treatments": [("Quench & Temper", 100)],
        "route": (
            "Wire drawing -> Cold heading (head form) -> Thread rolling -> "
            "Quench & temper -> Zinc/phosphate coating -> Inspection"
        ),
        "base_cost": 10.0, "base_lead": 7,
    },
    "bracket": {
        "materials": [
            ("Mild Steel (S235JR)", 60),
            ("Aluminum (6061-T6)", 26),
            ("Stainless Steel (AISI 304)", 14),
        ],
        "processes": [
            ("Sheet-Metal Forming", 50), ("CNC Milling", 35), ("Drilling & Tapping", 15),
        ],
        "route": (
            "Sheet blanking -> Bending/forming -> Drilling & tapping -> "
            "Deburring -> Powder coat/anodize -> Inspection"
        ),
        "base_cost": 25.0, "base_lead": 14,
    },
    "coupling": {
        "materials": [
            ("Carbon Steel (C45)", 60),
            ("Alloy Steel (42CrMo4)", 26),
            ("Ductile Iron (EN-GJS-500)", 14),
        ],
        "processes": [("Turning", 45), ("Broaching", 30), ("CNC Milling", 25)],
        "heat_treatments": [("Induction Hardening", 55), ("Quench & Temper", 45)],
        "route": (
            "Forged/bar blank -> Turning bore & OD -> Keyway broaching -> "
            "Drilling set-screw holes -> Heat treatment -> Balancing -> Inspection"
        ),
        "base_cost": 45.0, "base_lead": 21,
    },
    "flange": {
        "materials": [
            ("Carbon Steel (C45)", 58),
            ("Stainless Steel (AISI 316)", 26),
            ("Alloy Steel (42CrMo4)", 16),
        ],
        "processes": [("Forging", 40), ("Turning", 35), ("Drilling & Tapping", 25)],
        "route": (
            "Forging blank -> Rough facing & boring -> Bolt-hole drilling -> "
            "Finish facing (raised face) -> Surface finish -> NDT/Inspection"
        ),
        "base_cost": 55.0, "base_lead": 25,
    },
    "gear": {
        "materials": [
            ("Case-Hardening Steel (20MnCr5)", 60),
            ("Case-Hardening Steel (18CrNiMo7-6)", 24),
            ("Case-Hardening Steel (16MnCr5)", 11),
            ("Alloy Steel (42CrMo4)", 5),
        ],
        "processes": [
            ("Hobbing", 55), ("Gear Grinding", 20), ("Gear Shaping", 18), ("Broaching", 7),
        ],
        "heat_treatments": [
            ("Carburizing", 65), ("Induction Hardening", 20), ("Nitriding", 15),
        ],
        "route": (
            "Forged blank / bar stock -> Rough turning (OD/ID/face) -> "
            "Gear cutting (hobbing/shaping) -> Heat treatment (carburizing / "
            "induction hardening / nitriding) -> Finish grinding -> "
            "Inspection (tooth profile, lead, pitch)"
        ),
        "base_cost": 120.0, "base_lead": 35,
    },
    "nut": {
        "materials": [
            ("Property Class 8.8 Steel", 60),
            ("Stainless Steel (A2-70)", 25),
            ("Carbon Steel (C35)", 15),
        ],
        "processes": [("Cold Forming", 55), ("Drilling & Tapping", 45)],
        "heat_treatments": [("Quench & Temper", 100)],
        "route": (
            "Bar/wire -> Cold forming (blank) -> Piercing -> Tapping -> "
            "Heat treatment -> Surface coating -> Inspection"
        ),
        "base_cost": 8.0, "base_lead": 7,
    },
    "pulley": {
        "materials": [
            ("Gray Cast Iron (EN-GJL-250)", 60),
            ("Aluminum (6061-T6)", 26),
            ("Carbon Steel (C45)", 14),
        ],
        "processes": [("Casting", 40), ("Turning", 40), ("CNC Milling", 20)],
        "route": (
            "Casting / bar blank -> Turning bore & groove profile -> "
            "Keyway broaching -> Balancing -> Surface finish -> Inspection"
        ),
        "base_cost": 30.0, "base_lead": 18,
    },
    "screw": {
        "materials": [
            ("Property Class 12.9 Steel", 60),
            ("Stainless Steel (A2-70)", 26),
            ("Carbon Steel (C45)", 14),
        ],
        "processes": [("Cold Heading", 55), ("Thread Rolling", 45)],
        "heat_treatments": [("Quench & Temper", 100)],
        "route": (
            "Wire drawing -> Cold heading (head + recess) -> Thread rolling -> "
            "Quench & temper -> Surface coating -> Inspection"
        ),
        "base_cost": 12.0, "base_lead": 7,
    },
    "shaft": {
        "materials": [
            ("Alloy Steel (42CrMo4)", 58),
            ("Carbon Steel (C45)", 27),
            ("Stainless Steel (AISI 304)", 15),
        ],
        "processes": [
            ("Turning", 45), ("Cylindrical Grinding", 35), ("CNC Milling", 20),
        ],
        "heat_treatments": [("Induction Hardening", 60), ("Quench & Temper", 40)],
        "route": (
            "Bar stock -> Rough turning -> Milling keyways/splines -> "
            "Induction hardening -> Cylindrical grinding -> Inspection"
        ),
        "base_cost": 60.0, "base_lead": 21,
    },
}

# Fallback for a family with no dedicated profile (an unexpected folder name).
_DEFAULT_PROFILE: dict = {
    "materials": [
        ("Carbon Steel (C45)", 50),
        ("Aluminum (6061-T6)", 30),
        ("Stainless Steel (AISI 304)", 20),
    ],
    "processes": [("CNC Milling", 50), ("Turning", 30), ("Drilling & Tapping", 20)],
    "heat_treatments": None,
    "route": "Stock preparation -> Machining -> Heat treatment -> Finishing -> Inspection",
    "base_cost": 40.0, "base_lead": 14,
}

# The same grade appears in real PLM data as a trade name, a DIN/EN number, or
# a shop-floor code. These are the SAME material written differently -- not
# different materials -- so the normalizer can fold them onto one grade. This
# is the demonstrable point; a global pool of unrelated materials could not be.
_MATERIAL_ALIASES: dict[str, list[str]] = {
    "Mild Steel (S235JR)": ["S235JR", "1.0038", "St37-2"],
    "Carbon Steel (C35)": ["C35", "1.0501"],
    "Carbon Steel (C45)": ["C45", "1.0503", "AISI 1045"],
    "Alloy Steel (42CrMo4)": ["42CrMo4", "1.7225", "AISI 4140"],
    "Case-Hardening Steel (16MnCr5)": ["16MnCr5", "1.7131"],
    "Case-Hardening Steel (20MnCr5)": ["20MnCr5", "1.7147", "20 MnCr 5"],
    "Case-Hardening Steel (18CrNiMo7-6)": ["18CrNiMo7-6", "1.6587"],
    "Bearing Steel (100Cr6)": ["100Cr6", "1.3505"],
    "Chrome Steel (AISI 52100)": ["52100", "AISI 52100"],
    "Stainless Steel (AISI 304)": ["304", "1.4301", "X5CrNi18-10"],
    "Stainless Steel (AISI 316)": ["316", "1.4401"],
    "Stainless Steel (AISI 440C)": ["440C", "1.4125"],
    "Stainless Steel (A2-70)": ["A2-70", "304 (A2)"],
    "Aluminum (6061-T6)": ["6061-T6", "AA6061", "3.3211"],
    "Gray Cast Iron (EN-GJL-250)": ["EN-GJL-250", "GG25"],
    "Ductile Iron (EN-GJS-500)": ["EN-GJS-500", "GGG50"],
    "Property Class 8.8 Steel": ["Class 8.8", "Grade 8.8"],
    "Property Class 10.9 Steel": ["Class 10.9", "Grade 10.9"],
    "Property Class 12.9 Steel": ["Class 12.9", "Grade 12.9"],
}

# Placeholder tokens a records store carries when a grade is not yet decided.
_JUNK_MATERIAL_TOKENS = ["TBD", "SEE DRAWING"]

# Reverse lookup: any raw spelling (casefolded, and space-stripped) -> canonical.
_MATERIAL_ALIAS_LOOKUP: dict[str, str] = {}
for _canon, _aliases in _MATERIAL_ALIASES.items():
    for _form in (_canon, *_aliases):
        _MATERIAL_ALIAS_LOOKUP[_form.casefold()] = _canon
        _MATERIAL_ALIAS_LOOKUP[_form.casefold().replace(" ", "")] = _canon
_JUNK_SET = {t.casefold() for t in _JUNK_MATERIAL_TOKENS}

# Make/Buy and lifecycle are weighted so the exception looks like an exception:
# most parts are made in-house and active; a few are bought, discontinued, on a
# quality hold, or still a prototype.
_MAKE_OR_BUY = [("Make", 65), ("Buy", 35)]
_LIFECYCLE = [("Active", 85), ("Discontinued", 6), ("Quality hold", 5), ("Prototype", 4)]
# Anonymised supplier labels -- no real or realistic-looking company names.
_SUPPLIERS = [
    "Supplier A", "Supplier B", "Supplier C", "Supplier D", "Supplier E",
    "Supplier F", "In-House",
]
_LOT_SIZES = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
# Buy parts carry a procurement lead penalty over made-in-house parts.
_MAKE_OR_BUY_LEAD = {"Make": 1.0, "Buy": 1.6}


# ---------------------------------------------------------------------------
# Deterministic synthesis (no random, no dict-order dependence)
# ---------------------------------------------------------------------------


def _rng_int(file_id: str, key: str) -> int:
    """Return a stable non-negative integer for a (file_id, key) pair."""
    digest = hashlib.sha256(f"{file_id}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _choice(file_id: str, key: str, options: list):
    return options[_rng_int(file_id, key) % len(options)]


def _weighted_pick(file_id: str, key: str, options: list) -> str:
    """Deterministically pick one weighted option ``[(name, weight), ...]``."""
    total = sum(weight for _, weight in options)
    target = (_rng_int(file_id, key) % 1000) / 1000.0 * total
    cumulative = 0.0
    for name, weight in options:
        cumulative += weight
        if target < cumulative:
            return name
    return options[-1][0]


def _size_factor(file_id: str) -> float:
    """Return a deterministic per-part size factor in [0.6, 1.8].

    This is the ONLY component that varies part by part within a family. It is
    what makes similar parts spread around a family typical value, so the demo
    can say "this is an estimate, not a quote" -- there is a band, not a point.
    """
    return 0.6 + (_rng_int(file_id, "SizeFactor") % 1000) / 1000.0 * 1.2


def canonical_material(value):
    """Fold a messy/legacy material spelling onto its canonical grade.

    Trade names, DIN/EN numbers and shop codes ("1.7147", "20 MnCr 5") collapse
    onto the same grade; placeholder junk ("TBD", "SEE DRAWING") maps to None.
    Returned as the property the Context Layer normalizer demonstrates.
    """
    if value is None:
        return None
    text = " ".join(str(value).split())
    low = text.casefold()
    if low in _JUNK_SET:
        return None
    return (
        _MATERIAL_ALIAS_LOOKUP.get(low)
        or _MATERIAL_ALIAS_LOOKUP.get(low.replace(" ", ""))
        or text
    )


def _render_material(file_id: str, canonical: str) -> str:
    """Render a canonical grade as one of its real-world spellings.

    About 10% of records carry a placeholder ("TBD" / "SEE DRAWING") instead, so
    the normalizer has junk to drop. The rest are the canonical name or one of
    its DIN/EN / shop-code aliases -- the same grade, written differently.
    """
    if _rng_int(file_id, "MaterialJunk") % 100 < 10:
        return _choice(file_id, "MaterialJunkTok", _JUNK_MATERIAL_TOKENS)
    forms = [canonical, *_MATERIAL_ALIASES.get(canonical, [])]
    return forms[_rng_int(file_id, "MaterialForm") % len(forms)]


def _profile_for(family: str) -> dict:
    return _FAMILY_PROFILES.get(family.strip().lower(), _DEFAULT_PROFILE)


def synthesize_context(file_id: str, family: str) -> Optional[dict]:
    """Return a synthetic context for *file_id*, or ``None`` for the empty bucket.

    Every value is a pure function of *file_id* (and the CSV-derived *family*),
    so the output is reproducible. Values are drawn from the *family*'s own
    palette, and the numeric fields come from a cost/lead equation rather than a
    uniform random number, so parts in one family agree on their material,
    process and route and cluster around a family-typical cost.
    """
    bucket = _rng_int(file_id, "__bucket__") % 100
    if bucket < 20:
        return None  # ~20% have no record at all

    profile = _profile_for(family)
    canonical = _weighted_pick(file_id, "Material", profile["materials"])
    process = _weighted_pick(file_id, "Process", profile["processes"])
    make_or_buy = _weighted_pick(file_id, "MakeOrBuy", _MAKE_OR_BUY)

    context: dict = {"PartFamily": family}
    context["Material"] = _render_material(file_id, canonical)
    context["Process"] = process
    heat_treatments = profile.get("heat_treatments")
    if heat_treatments:
        context["HeatTreatment"] = _weighted_pick(file_id, "HeatTreatment", heat_treatments)
    context["ProcessRoute"] = profile["route"]
    context["MakeOrBuy"] = make_or_buy
    context["Supplier"] = _choice(file_id, "Supplier", _SUPPLIERS)
    context["LifecycleStatus"] = _weighted_pick(file_id, "LifecycleStatus", _LIFECYCLE)

    if bucket >= 50:  # ~50% get the numeric fields too
        size = _size_factor(file_id)
        mat_mult = _MATERIAL_MULTIPLIER.get(canonical, 1.0)
        proc_mult = _PROCESS_MULTIPLIER.get(process, 1.0)
        unit_cost = profile["base_cost"] * mat_mult * proc_mult * size
        context["UnitCost"] = round(unit_cost, 2)
        lead = profile["base_lead"] * (0.6 + 0.4 * proc_mult) * _MAKE_OR_BUY_LEAD[make_or_buy]
        context["LeadTimeDays"] = max(1, round(lead))
        context["LotSize"] = _choice(file_id, "LotSize", _LOT_SIZES)

    return context


def default_units(currency: str = "USD") -> dict:
    """Return the unit labels for the numeric fields this tool writes."""
    return {"UnitCost": currency, "LeadTimeDays": "days", "LotSize": "pcs"}


# ---------------------------------------------------------------------------
# Stage A: scan a CAD tree into a families CSV
# ---------------------------------------------------------------------------


def sha256_of(path: pathlib.Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _family_of(root: pathlib.Path, path: pathlib.Path) -> str:
    """Return the first path component of *path* under *root* (its family)."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "_root"


def scan_source_tree(source_root: pathlib.Path) -> list[tuple[str, str, str]]:
    """Return ``[(file_id, family, source_path), ...]`` sorted by (id, path).

    Every regular file under *source_root* is hashed. The family is taken from
    the directory layout, but keyed by the content hash, so two copies of the
    same file under different families produce two rows with the same id.
    """
    rows: list[tuple[str, str, str]] = []
    for dirpath, _dirs, files in os.walk(source_root):
        for name in files:
            path = pathlib.Path(dirpath) / name
            if not path.is_file():
                continue
            try:
                file_id = sha256_of(path)
            except OSError as exc:
                print(f"  skipped (unreadable): {path} ({exc})", file=sys.stderr)
                continue
            family = _family_of(source_root, path)
            rows.append((file_id, family, str(path)))
    rows.sort(key=lambda r: (r[0], r[2]))
    return rows


def cross_family_count(rows: Iterable[tuple[str, str, str]]) -> int:
    """Return how many file_ids appear under more than one family."""
    families: dict[str, set[str]] = {}
    for file_id, family, _path in rows:
        families.setdefault(file_id, set()).add(family)
    return sum(1 for fams in families.values() if len(fams) > 1)


def write_families_csv(rows: list[tuple[str, str, str]], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_id", "family", "source_path"])
        writer.writerows(rows)


def load_families_csv(path: pathlib.Path) -> list[tuple[str, str, str]]:
    """Read a families CSV back into ``[(file_id, family, source_path), ...]``."""
    rows: list[tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != ["file_id", "family", "source_path"]:
            raise ValueError(
                f"{path} is not a families CSV (unexpected header {header!r})."
            )
        for row in reader:
            if len(row) < 3:
                continue
            rows.append((row[0].strip(), row[1].strip(), row[2]))
    return rows


# ---------------------------------------------------------------------------
# Stage B: build context.json from the CSV and an index's ids
# ---------------------------------------------------------------------------


def _pick_family(families: set[str]) -> str:
    """Deterministically pick one family for a cross-family id (lexicographic min)."""
    return sorted(families)[0]


def build_context_document(
    rows: Iterable[tuple[str, str, str]],
    index_ids: Iterable[str],
    currency: str = "USD",
) -> tuple[dict, dict]:
    """Build the context.json document (and stats) for the given index ids.

    Pure and SDK-free: takes the CSV rows and the index's id set, returns
    ``(document, stats)``. The document is deterministic -- no timestamps in the
    per-part records -- so two runs on the same inputs are byte-identical. The
    top-level ``units`` records the unit of each numeric field (currency for
    ``UnitCost``) so a client can label the numbers it shows.
    """
    families_by_id: dict[str, set[str]] = {}
    for file_id, family, _path in rows:
        families_by_id.setdefault(file_id, set()).add(family)

    parts: dict[str, dict] = {}
    stats = {"seeded": 0, "empty_bucket": 0, "not_in_csv": 0, "cross_family": 0}
    stats["cross_family"] = sum(
        1 for fams in families_by_id.values() if len(fams) > 1
    )

    for file_id in sorted(set(str(i) for i in index_ids)):
        families = families_by_id.get(file_id)
        if not families:
            stats["not_in_csv"] += 1
            continue
        family = _pick_family(families)
        context = synthesize_context(file_id, family)
        if context is None:
            stats["empty_bucket"] += 1
            continue
        parts[file_id] = {"context": context, "source": SYNTHETIC_SOURCE}
        stats["seeded"] += 1

    document = {
        "version": 1,
        "updated_at": None,
        "note": f"{SYNTHETIC_SOURCE} generated by tools/seed_context_store.py",
        "units": default_units(currency),
        "parts": parts,
    }
    return document, stats


def serialize_document(document: dict) -> bytes:
    """Serialize a context document deterministically (byte-stable)."""
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_scan(args: argparse.Namespace) -> int:
    source_root = pathlib.Path(args.source).expanduser().resolve()
    if not source_root.is_dir():
        print(f"Source tree not found: {source_root}", file=sys.stderr)
        return 2
    rows = scan_source_tree(source_root)
    write_families_csv(rows, pathlib.Path(args.out).expanduser())
    print(f"Scanned:      {len(rows)} files under {source_root}")
    print(f"Distinct ids: {len({r[0] for r in rows})}")
    print(f"Cross-family: {cross_family_count(rows)} ids under more than one family")
    print(f"Wrote:        {args.out}")
    return 0


def _cmd_seed(args: argparse.Namespace) -> int:
    families_path = pathlib.Path(args.families).expanduser()
    if not families_path.is_file():
        print(f"Families CSV not found: {families_path}", file=sys.stderr)
        return 2

    import core  # noqa: E402 - only stage B needs the index, and it needs the SDK
    import index_registry  # noqa: E402 - only stage B needs the index, and it needs the SDK
    import config  # noqa: E402

    try:
        core._validate_index_name(args.index)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    context_path = config.INDEXES_DIR / args.index / "context.json"
    if context_path.exists() and not (args.force or args.dry_run):
        print(
            f"{context_path} already exists. Re-run with --force to replace it "
            "or --dry-run to preview.",
            file=sys.stderr,
        )
        return 2

    try:
        config.load_env_file()
        index_ids = index_registry._registered_index_ids(args.index)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rows = load_families_csv(families_path)
    document, stats = build_context_document(rows, index_ids, currency=args.currency)

    print(f"Index:        {args.index} ({len(index_ids)} registered ids)")
    print(f"CSV rows:     {len(rows)}")
    print(f"Seeded:       {stats['seeded']} parts")
    print(f"Empty bucket: {stats['empty_bucket']} (intentionally no record)")
    print(f"Not in CSV:   {stats['not_in_csv']} registered ids with no source file")
    print(f"Cross-family: {stats['cross_family']} ids under more than one family")

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    payload = serialize_document(document)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_bytes(payload)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\nWrote {context_path} ({len(document['parts'])} parts) at {stamp}.")
    print(f"({SYNTHETIC_SOURCE}; the server records the marker on every part.)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Stage A: hash a CAD tree into a families CSV")
    scan.add_argument("--source", required=True, help="Root of the CAD source tree")
    scan.add_argument("--out", required=True, help="Output CSV path")
    scan.set_defaults(func=_cmd_scan)

    seed = sub.add_parser("seed", help="Stage B: write context.json for an index")
    seed.add_argument("--families", required=True, help="Families CSV from 'scan'")
    seed.add_argument("--index", required=True, help="Target index name")
    seed.add_argument(
        "--dry-run", action="store_true", help="Report what would change and write nothing"
    )
    seed.add_argument(
        "--force", action="store_true", help="Replace an existing context.json"
    )
    seed.add_argument(
        "--currency", default="USD",
        help="Currency label recorded as the unit of UnitCost (default USD)",
    )
    seed.set_defaults(func=_cmd_seed)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
