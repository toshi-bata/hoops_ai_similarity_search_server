"""Tests for tools/seed_context_store.py (SDK-free).

The seed tool fabricates sample production records. The two properties that
matter are determinism (re-running must produce a byte-identical file) and
that the family is decided by the content hash, not the file name. The stage-B
``core`` import is lazy, so importing the module here needs no SDK.
"""

import pathlib
import statistics
import sys
import tempfile
import unittest
from collections import Counter

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "tools"))

import seed_context_store as seed  # noqa: E402


def _ids(n):
    return [f"{i:064d}" for i in range(n)]


class TestSynthesisDeterminism(unittest.TestCase):
    def test_same_inputs_same_output(self):
        a = seed.synthesize_context("abc", "gear")
        b = seed.synthesize_context("abc", "gear")
        self.assertEqual(a, b)

    def test_distribution_has_a_mix(self):
        ids = _ids(400)
        results = [seed.synthesize_context(i, "gear") for i in ids]
        empties = [r for r in results if r is None]
        complete = [r for r in results if r and "UnitCost" in r]
        strings_only = [r for r in results if r and "UnitCost" not in r]
        # Not exact ratios, just proof each bucket is exercised.
        self.assertTrue(empties)
        self.assertTrue(complete)
        self.assertTrue(strings_only)

    def test_family_flows_into_partfamily(self):
        fid = next(i for i in _ids(50) if seed.synthesize_context(i, "gear"))
        ctx = seed.synthesize_context(fid, "gear")
        self.assertEqual(ctx["PartFamily"], "gear")


class TestCrossFamily(unittest.TestCase):
    def test_pick_family_is_min(self):
        self.assertEqual(seed._pick_family({"nut", "bolt", "gear"}), "bolt")

    def test_cross_family_counted_and_picked(self):
        fid = next(i for i in _ids(50) if seed.synthesize_context(i, "aaa"))
        rows = [
            (fid, "zzz", "/cad/zzz/part.stp"),
            (fid, "aaa", "/cad/aaa/part.stp"),
            ("other", "bbb", "/cad/bbb/x.stp"),
        ]
        doc, stats = seed.build_context_document(rows, [fid, "other"])
        self.assertEqual(stats["cross_family"], 1)
        # Lexicographic min family wins.
        self.assertEqual(doc["parts"][fid]["context"]["PartFamily"], "aaa")


class TestBuildDeterminism(unittest.TestCase):
    def test_byte_identical_regardless_of_id_order(self):
        rows = [(f"{i:064d}", "gear", f"/cad/gear/{i}.stp") for i in range(30)]
        ids = [r[0] for r in rows]
        doc1, _ = seed.build_context_document(rows, ids)
        doc2, _ = seed.build_context_document(rows, list(reversed(ids)))
        self.assertEqual(
            seed.serialize_document(doc1), seed.serialize_document(doc2)
        )

    def test_records_carry_synthetic_marker(self):
        rows = [(f"{i:064d}", "gear", f"/cad/gear/{i}.stp") for i in range(30)]
        doc, _ = seed.build_context_document(rows, [r[0] for r in rows])
        some = next(iter(doc["parts"].values()))
        self.assertEqual(some["source"], seed.SYNTHETIC_SOURCE)

    def test_registered_id_without_source_reported(self):
        rows = [("known", "gear", "/cad/gear/a.stp")]
        _doc, stats = seed.build_context_document(rows, ["known", "missing"])
        self.assertEqual(stats["not_in_csv"], 1)


class TestScan(unittest.TestCase):
    def test_same_content_two_families_is_one_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "famA").mkdir()
            (root / "famB").mkdir()
            (root / "famA" / "part.stp").write_bytes(b"SAME BYTES")
            (root / "famB" / "part.stp").write_bytes(b"SAME BYTES")
            (root / "famA" / "unique.stp").write_bytes(b"DIFFERENT")

            rows = seed.scan_source_tree(root)
            self.assertEqual(len(rows), 3)
            self.assertEqual(seed.cross_family_count(rows), 1)
            # Rows are sorted by (file_id, source_path).
            self.assertEqual(rows, sorted(rows, key=lambda r: (r[0], r[2])))

    def test_roundtrip_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "gear").mkdir()
            (root / "gear" / "a.stp").write_bytes(b"A")
            rows = seed.scan_source_tree(root)
            out = root / "families.csv"
            seed.write_families_csv(rows, out)
            self.assertEqual(seed.load_families_csv(out), rows)


class TestFamilyStatistics(unittest.TestCase):
    """Lock in the properties that make the prediction meaningful.

    If these fail, the seeded data is family-independent noise again and any UI
    built on top of it shows random numbers. See STEP 1-4 of the task.
    """

    FAMILIES = [
        "bearing", "bolt", "bracket", "coupling", "flange",
        "gear", "nut", "pulley", "screw", "shaft",
    ]
    SAMPLE = 800

    def _by_family(self):
        out = {}
        for fam in self.FAMILIES:
            records = []
            for i in range(self.SAMPLE):
                ctx = seed.synthesize_context(f"{fam}-{i}", fam)
                if ctx:
                    records.append(ctx)
            out[fam] = records
        return out

    def test_families_are_well_separated(self):
        # Family-mean UnitCost max/min >= 3.0.
        data = self._by_family()
        means = {}
        for fam, recs in data.items():
            costs = [r["UnitCost"] for r in recs if "UnitCost" in r]
            means[fam] = statistics.fmean(costs)
        ratio = max(means.values()) / min(means.values())
        self.assertGreaterEqual(ratio, 3.0, f"family cost means too close: {means}")

    def test_within_family_is_tight(self):
        # Each family's UnitCost coefficient of variation (std/mean) <= 0.4.
        data = self._by_family()
        for fam, recs in data.items():
            costs = [r["UnitCost"] for r in recs if "UnitCost" in r]
            mean = statistics.fmean(costs)
            cv = statistics.pstdev(costs) / mean
            self.assertLessEqual(cv, 0.4, f"{fam} UnitCost CV {cv:.3f} > 0.4")

    def test_material_converges_per_family(self):
        # The most common normalized Material is >= 50% of a family's records.
        data = self._by_family()
        for fam, recs in data.items():
            mats = [seed.canonical_material(r["Material"]) for r in recs]
            mats = [m for m in mats if m]
            top = Counter(mats).most_common(1)[0]
            share = top[1] / len(mats)
            self.assertGreaterEqual(
                share, 0.5, f"{fam} top material only {share:.2f} ({top[0]})"
            )

    def test_process_route_is_unique_per_family(self):
        data = self._by_family()
        for fam, recs in data.items():
            routes = {r["ProcessRoute"] for r in recs}
            self.assertEqual(len(routes), 1, f"{fam} has {len(routes)} routes")


class TestMaterialNormalization(unittest.TestCase):
    def test_aliases_fold_to_one_grade(self):
        self.assertEqual(
            seed.canonical_material("1.7147"),
            seed.canonical_material("20 MnCr 5"),
        )
        self.assertEqual(
            seed.canonical_material("20MnCr5"),
            "Case-Hardening Steel (20MnCr5)",
        )

    def test_junk_maps_to_none(self):
        self.assertIsNone(seed.canonical_material("TBD"))
        self.assertIsNone(seed.canonical_material("SEE DRAWING"))


class TestUnits(unittest.TestCase):
    def test_units_present_and_currency_flows(self):
        rows = [(f"{i:064d}", "gear", f"/cad/gear/{i}.stp") for i in range(20)]
        ids = [r[0] for r in rows]
        doc, _ = seed.build_context_document(rows, ids, currency="EUR")
        self.assertEqual(doc["units"]["UnitCost"], "EUR")
        self.assertEqual(doc["units"]["LeadTimeDays"], "days")
        self.assertEqual(doc["units"]["LotSize"], "pcs")

    def test_currency_defaults_to_usd(self):
        rows = [(f"{i:064d}", "bolt", f"/cad/bolt/{i}.stp") for i in range(5)]
        doc, _ = seed.build_context_document(rows, [r[0] for r in rows])
        self.assertEqual(doc["units"]["UnitCost"], "USD")


if __name__ == "__main__":
    unittest.main()
