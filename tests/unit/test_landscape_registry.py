"""Unit tests for the Stage 5 landscape registry + split contract.

No Graphab. Synthetic specs use tmp_path with empty placeholder
files for the required artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from habconn.experiments.landscape_registry import (
    SMALL_VECTOR_001_ID,
    LandscapeSpec,
    LandscapeSplit,
    builtin_landscape_specs,
    development_split,
    split_is_transfer_ready,
    split_summary,
    validate_landscape_spec,
    validate_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "candidates.shp").write_bytes(b"")
    (d / "habitat.tif").write_bytes(b"")
    (d / "resistance.tif").write_bytes(b"")
    return d


def _make_spec(tmp_path: Path, landscape_id: str = "synthetic_001") -> LandscapeSpec:
    d = _make_synthetic_dir(tmp_path, landscape_id)
    return LandscapeSpec(landscape_id=landscape_id, data_dir=d)


# ---------------------------------------------------------------------------
# LandscapeSpec
# ---------------------------------------------------------------------------


class TestLandscapeSpec:
    def test_string_data_dir_coerced_to_path(self, tmp_path):
        d = _make_synthetic_dir(tmp_path, "ok")
        spec = LandscapeSpec(landscape_id="ok", data_dir=str(d))
        assert isinstance(spec.data_dir, Path)
        assert spec.data_dir == d

    def test_required_files_resolve_under_data_dir(self, tmp_path):
        spec = _make_spec(tmp_path, "ok")
        assert spec.candidates_path.name == "candidates.shp"
        assert spec.habitat_raster_path.name == "habitat.tif"
        assert spec.resistance_raster_path.name == "resistance.tif"
        for p in spec.required_files():
            assert p.parent == spec.data_dir

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            "   ",
            "..",
            ".",
            "../escape",
            "x/y",
            "x\\y",
            "/abs/path",
            "x..y",
            ".hidden",
            "-flag",
            "name with spaces",
            "name@punct",
        ],
    )
    def test_invalid_landscape_id_rejected(self, tmp_path, bad_id):
        with pytest.raises((ValueError, TypeError)):
            LandscapeSpec(landscape_id=bad_id, data_dir=tmp_path)

    def test_non_string_landscape_id_rejected(self, tmp_path):
        with pytest.raises(TypeError):
            LandscapeSpec(landscape_id=123, data_dir=tmp_path)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "good_id",
        [
            "small_vector_001",
            "site-a",
            "v1.0",
            "Region_001",
            "_internal_fixture",
            "0_first",
        ],
    )
    def test_valid_landscape_id_accepted(self, tmp_path, good_id):
        d = _make_synthetic_dir(tmp_path, good_id)
        spec = LandscapeSpec(landscape_id=good_id, data_dir=d)
        assert spec.landscape_id == good_id


class TestValidateLandscapeSpec:
    def test_present_files_pass(self, tmp_path):
        spec = _make_spec(tmp_path, "ok")
        validate_landscape_spec(spec)

    def test_missing_files_reported(self, tmp_path):
        d = tmp_path / "incomplete"
        d.mkdir()
        spec = LandscapeSpec(landscape_id="incomplete", data_dir=d)
        with pytest.raises(FileNotFoundError) as excinfo:
            validate_landscape_spec(spec)
        msg = str(excinfo.value)
        assert "candidates.shp" in msg
        assert "habitat.tif" in msg
        assert "resistance.tif" in msg
        assert "incomplete" in msg

    def test_partial_missing_reports_only_missing(self, tmp_path):
        d = tmp_path / "partial"
        d.mkdir()
        (d / "candidates.shp").write_bytes(b"")
        # habitat + resistance intentionally absent
        spec = LandscapeSpec(landscape_id="partial", data_dir=d)
        with pytest.raises(FileNotFoundError) as excinfo:
            validate_landscape_spec(spec)
        msg = str(excinfo.value)
        assert "candidates.shp" not in msg
        assert "habitat.tif" in msg
        assert "resistance.tif" in msg


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestBuiltinRegistry:
    def test_contains_only_small_vector_001(self):
        registry = builtin_landscape_specs()
        assert set(registry.keys()) == {SMALL_VECTOR_001_ID}

    def test_small_vector_001_is_development_fixture(self):
        registry = builtin_landscape_specs()
        spec = registry[SMALL_VECTOR_001_ID]
        assert spec.is_development_fixture is True

    def test_registry_resolves_under_supplied_package_root(self, tmp_path):
        # Use a synthetic package root just to prove the helper threads
        # ``package_root`` through.
        registry = builtin_landscape_specs(package_root=tmp_path)
        spec = registry[SMALL_VECTOR_001_ID]
        assert spec.data_dir == tmp_path / "data" / "examples" / SMALL_VECTOR_001_ID


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


class TestDevelopmentSplit:
    def test_validates_against_builtin_registry(self):
        registry = builtin_landscape_specs()
        split = development_split()
        validate_split(split, registry)

    def test_is_not_transfer_ready(self):
        assert split_is_transfer_ready(development_split()) is False

    def test_summary_records_no_held_out(self):
        registry = builtin_landscape_specs()
        s = split_summary(development_split(), registry)
        assert s["n_train"] == 1
        assert s["n_validation"] == 0
        assert s["n_test"] == 0
        assert s["is_transfer_ready"] is False
        assert SMALL_VECTOR_001_ID in s["train_landscape_ids"]
        # JSON-serializable
        json.dumps(s)


class TestValidateSplit:
    def test_empty_split_rejected(self):
        registry = {"a": object()}  # type: ignore[dict-item]
        with pytest.raises(ValueError):
            validate_split(LandscapeSplit(split_name="empty"), registry)  # type: ignore[arg-type]

    def test_overlap_train_validation_rejected(self):
        registry = {
            lid: object()  # type: ignore[dict-item]
            for lid in ("a", "b")
        }
        split = LandscapeSplit(
            split_name="bad",
            train_landscape_ids=("a", "b"),
            validation_landscape_ids=("b",),
        )
        with pytest.raises(ValueError) as excinfo:
            validate_split(split, registry)
        assert "train ∩ validation" in str(excinfo.value)

    def test_overlap_train_test_rejected(self):
        registry = {lid: object() for lid in ("a", "b")}  # type: ignore[dict-item]
        split = LandscapeSplit(
            split_name="bad",
            train_landscape_ids=("a",),
            test_landscape_ids=("a",),
        )
        with pytest.raises(ValueError) as excinfo:
            validate_split(split, registry)
        assert "train ∩ test" in str(excinfo.value)

    def test_overlap_validation_test_rejected(self):
        registry = {lid: object() for lid in ("a", "b")}  # type: ignore[dict-item]
        split = LandscapeSplit(
            split_name="bad",
            train_landscape_ids=("a",),
            validation_landscape_ids=("b",),
            test_landscape_ids=("b",),
        )
        with pytest.raises(ValueError) as excinfo:
            validate_split(split, registry)
        assert "validation ∩ test" in str(excinfo.value)

    def test_unknown_landscape_id_rejected(self):
        registry = {"a": object()}  # type: ignore[dict-item]
        split = LandscapeSplit(
            split_name="bad",
            train_landscape_ids=("a",),
            test_landscape_ids=("nonexistent",),
        )
        with pytest.raises(ValueError) as excinfo:
            validate_split(split, registry)
        assert "nonexistent" in str(excinfo.value)

    def test_iterable_inputs_coerced_to_tuple(self):
        # Construct from lists; the dataclass should normalize to tuple.
        split = LandscapeSplit(
            split_name="ok",
            train_landscape_ids=["a", "b"],  # type: ignore[arg-type]
            validation_landscape_ids=["c"],  # type: ignore[arg-type]
        )
        assert split.train_landscape_ids == ("a", "b")
        assert split.validation_landscape_ids == ("c",)


class TestSplitIsTransferReady:
    def test_train_plus_validation_only(self):
        split = LandscapeSplit(
            split_name="ok",
            train_landscape_ids=("a", "b"),
            validation_landscape_ids=("c",),
        )
        assert split_is_transfer_ready(split) is True

    def test_train_plus_test_only(self):
        split = LandscapeSplit(
            split_name="ok",
            train_landscape_ids=("a",),
            test_landscape_ids=("b",),
        )
        assert split_is_transfer_ready(split) is True

    def test_train_only_is_not_transfer_ready(self):
        split = LandscapeSplit(
            split_name="ok",
            train_landscape_ids=("a", "b"),
        )
        assert split_is_transfer_ready(split) is False

    def test_held_out_only_is_not_transfer_ready(self):
        split = LandscapeSplit(
            split_name="ok",
            validation_landscape_ids=("a",),
            test_landscape_ids=("b",),
        )
        assert split_is_transfer_ready(split) is False


class TestSplitSummary:
    def test_payload_shape(self):
        registry = {
            "a": object(),  # type: ignore[dict-item]
            "b": object(),  # type: ignore[dict-item]
            "c": object(),  # type: ignore[dict-item]
        }
        split = LandscapeSplit(
            split_name="ok",
            train_landscape_ids=("a",),
            validation_landscape_ids=("b",),
            test_landscape_ids=("c",),
            notes="example",
        )
        s = split_summary(split, registry)
        assert s["split_name"] == "ok"
        assert s["train_landscape_ids"] == ["a"]
        assert s["validation_landscape_ids"] == ["b"]
        assert s["test_landscape_ids"] == ["c"]
        assert s["n_train"] == 1
        assert s["n_validation"] == 1
        assert s["n_test"] == 1
        assert s["is_transfer_ready"] is True
        assert s["registry_size"] == 3
        assert s["notes"] == "example"
        json.dumps(s)


# ---------------------------------------------------------------------------
# Bundled-fixture contract test
# ---------------------------------------------------------------------------


class TestBundledFixtureContract:
    """The built-in registry must point at a real bundled fixture."""

    def test_small_vector_001_files_exist(self):
        registry = builtin_landscape_specs()
        spec = registry[SMALL_VECTOR_001_ID]
        for p in spec.required_files():
            assert p.exists(), f"Bundled fixture file missing: {p}"

    def test_validate_landscape_spec_passes_on_bundled_fixture(self):
        registry = builtin_landscape_specs()
        validate_landscape_spec(registry[SMALL_VECTOR_001_ID])
