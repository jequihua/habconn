"""Landscape registry and split contract.

Stage 5 foundation: register the bundled landscapes the package
knows about and represent landscape-level splits explicitly so a
later transfer-learning milestone has a concrete data contract to
build on.

This module deliberately ships **only the bundled development
fixture** (`small_vector_001`). It does not synthesize landscapes,
does not generate splits, and does not enable transfer learning.

Public surface:

- `LandscapeSpec` — one bundled landscape (id, data dir, file
  layout, optional development-fixture flag).
- `LandscapeSplit` — landscape-level train / validation / test
  split (immutable tuples of landscape ids).
- `builtin_landscape_specs(package_root)` — the registry of bundled
  landscapes; currently `{ "small_vector_001": ... }`.
- `development_split()` — the single-landscape split used while the
  registry still has only the development fixture; explicitly
  **not** transfer-learning evidence.
- `validate_landscape_spec(spec)` — file-existence and id checks.
- `validate_split(split, registry)` — split integrity:
  pairwise-disjoint sets, every id present, slug-safe ids.
- `split_is_transfer_ready(split)` — True iff there is at least one
  train id and at least one held-out validation or test id.
- `split_summary(split, registry)` — JSON-serializable dict view.

Slug rule for `landscape_id`: ``^[A-Za-z0-9_][A-Za-z0-9_.-]*$``,
no `/` or `\\`, no `..` substring, not absolute or
drive-qualified. Matches the existing run-name slug rule used by
the experiment contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional


_LANDSCAPE_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def _validate_landscape_id(landscape_id: object) -> str:
    """Slug-style validation. Returns the validated id."""
    if not isinstance(landscape_id, str):
        raise TypeError(
            f"landscape_id must be a string, got {type(landscape_id).__name__}"
        )
    if not landscape_id or not landscape_id.strip():
        raise ValueError("landscape_id must be a non-empty, non-whitespace string")
    if landscape_id in (".", ".."):
        raise ValueError(f"landscape_id {landscape_id!r} is reserved")
    if "/" in landscape_id or "\\" in landscape_id:
        raise ValueError(
            f"landscape_id must not contain path separators: {landscape_id!r}"
        )
    if ".." in landscape_id:
        raise ValueError(
            f"landscape_id must not contain '..' (path traversal): {landscape_id!r}"
        )
    if Path(landscape_id).is_absolute() or (
        len(landscape_id) >= 2 and landscape_id[1] == ":"
    ):
        raise ValueError(
            f"landscape_id must not be an absolute path: {landscape_id!r}"
        )
    if not _LANDSCAPE_ID_RE.match(landscape_id):
        raise ValueError(
            f"landscape_id {landscape_id!r} must match "
            f"[A-Za-z0-9_][A-Za-z0-9_.-]* (letters, digits, underscore, "
            f"hyphen, dot; not starting with '.' or '-')"
        )
    return landscape_id


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandscapeSpec:
    """Static description of one bundled landscape.

    The three filenames are resolved relative to ``data_dir``.
    ``is_development_fixture`` marks landscapes that exist only to
    drive the smoke-test path; downstream callers should refuse to
    treat such landscapes as transfer-learning evidence.
    """

    landscape_id: str
    data_dir: Path
    candidates_filename: str = "candidates.shp"
    habitat_filename: str = "habitat.tif"
    resistance_filename: str = "resistance.tif"
    id_column: str = "lyr_1"
    area_column: str = "area"
    is_development_fixture: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks normal assignment; use object.__setattr__
        # for the path coercion and the id validation.
        _validate_landscape_id(self.landscape_id)
        object.__setattr__(self, "data_dir", Path(self.data_dir))

    # Convenience helpers ---------------------------------------------------

    @property
    def candidates_path(self) -> Path:
        return self.data_dir / self.candidates_filename

    @property
    def habitat_raster_path(self) -> Path:
        return self.data_dir / self.habitat_filename

    @property
    def resistance_raster_path(self) -> Path:
        return self.data_dir / self.resistance_filename

    def required_files(self) -> tuple[Path, ...]:
        return (
            self.candidates_path,
            self.habitat_raster_path,
            self.resistance_raster_path,
        )


def validate_landscape_spec(spec: LandscapeSpec) -> None:
    """Verify the three required landscape files exist on disk.

    Raises ``FileNotFoundError`` with the missing path(s). Does not
    open the files; that's the problem-loader's job.
    """
    missing = [str(p) for p in spec.required_files() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Landscape {spec.landscape_id!r}: missing required file(s): "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LandscapeSplit:
    """Landscape-level train / validation / test split.

    Landscape-level (not planning-unit-level) on purpose: transfer
    learning evaluates a policy on landscapes it has not trained on,
    so the split unit must be the landscape.
    """

    split_name: str
    train_landscape_ids: tuple[str, ...] = ()
    validation_landscape_ids: tuple[str, ...] = ()
    test_landscape_ids: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.split_name, str) or not self.split_name.strip():
            raise ValueError("split_name must be a non-empty string")
        # Normalize each set to tuple so equality is stable.
        for field_name in (
            "train_landscape_ids",
            "validation_landscape_ids",
            "test_landscape_ids",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                object.__setattr__(self, field_name, tuple(value))
            # Validate each id.
            for lid in getattr(self, field_name):
                _validate_landscape_id(lid)

    def all_landscape_ids(self) -> tuple[str, ...]:
        return (
            self.train_landscape_ids
            + self.validation_landscape_ids
            + self.test_landscape_ids
        )

    def held_out_landscape_ids(self) -> tuple[str, ...]:
        return self.validation_landscape_ids + self.test_landscape_ids


def validate_split(
    split: LandscapeSplit, registry: Mapping[str, LandscapeSpec]
) -> None:
    """Reject overlap, unknown ids, or an empty split."""
    train = set(split.train_landscape_ids)
    val = set(split.validation_landscape_ids)
    test = set(split.test_landscape_ids)

    if not (train | val | test):
        raise ValueError(
            f"Split {split.split_name!r} contains no landscape ids"
        )

    overlap_train_val = train & val
    overlap_train_test = train & test
    overlap_val_test = val & test
    if overlap_train_val:
        raise ValueError(
            f"Split {split.split_name!r}: train ∩ validation overlap: "
            f"{sorted(overlap_train_val)}"
        )
    if overlap_train_test:
        raise ValueError(
            f"Split {split.split_name!r}: train ∩ test overlap: "
            f"{sorted(overlap_train_test)}"
        )
    if overlap_val_test:
        raise ValueError(
            f"Split {split.split_name!r}: validation ∩ test overlap: "
            f"{sorted(overlap_val_test)}"
        )

    unknown = [lid for lid in split.all_landscape_ids() if lid not in registry]
    if unknown:
        raise ValueError(
            f"Split {split.split_name!r} references unknown landscape ids: "
            f"{sorted(set(unknown))}"
        )


def split_is_transfer_ready(split: LandscapeSplit) -> bool:
    """True iff there is ≥1 train id and ≥1 held-out (val or test) id.

    Does not require a registry: a split with empty train and empty
    held-out sets is not transfer-ready regardless of what the
    registry contains.
    """
    return bool(split.train_landscape_ids) and bool(split.held_out_landscape_ids())


def split_summary(
    split: LandscapeSplit, registry: Mapping[str, LandscapeSpec]
) -> dict:
    """JSON-serializable view of the split for run-summary artifacts."""
    return {
        "split_name": split.split_name,
        "train_landscape_ids": list(split.train_landscape_ids),
        "validation_landscape_ids": list(split.validation_landscape_ids),
        "test_landscape_ids": list(split.test_landscape_ids),
        "n_train": len(split.train_landscape_ids),
        "n_validation": len(split.validation_landscape_ids),
        "n_test": len(split.test_landscape_ids),
        "is_transfer_ready": split_is_transfer_ready(split),
        "registry_size": len(registry),
        "notes": split.notes,
    }


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------


SMALL_VECTOR_001_ID = "small_vector_001"


def _default_package_root() -> Path:
    """Resolve ``08_pkg/habconn/`` from this module's location.

    The module lives at
    ``08_pkg/habconn/src/habconn/experiments/landscape_registry.py``,
    so the package root is two levels above ``src/``.
    """
    return Path(__file__).resolve().parents[3]


def builtin_landscape_specs(
    package_root: Optional[Path] = None,
) -> dict[str, LandscapeSpec]:
    """Return the registry of bundled landscapes.

    Today this contains only the development fixture
    ``small_vector_001``. Adding a new landscape requires a separate
    milestone with real data and split definitions.
    """
    root = Path(package_root) if package_root is not None else _default_package_root()
    return {
        SMALL_VECTOR_001_ID: LandscapeSpec(
            landscape_id=SMALL_VECTOR_001_ID,
            data_dir=root / "data" / "examples" / SMALL_VECTOR_001_ID,
            candidates_filename="candidates.shp",
            habitat_filename="habitat.tif",
            resistance_filename="resistance.tif",
            id_column="lyr_1",
            area_column="area",
            is_development_fixture=True,
            notes=(
                "Bundled development fixture used by the Stage 4 single-"
                "landscape DRL workflow. Not transfer-learning evidence."
            ),
        ),
    }


def development_split() -> LandscapeSplit:
    """The single-landscape split that names ``small_vector_001``.

    This split is intentionally **not** transfer-ready:
    ``split_is_transfer_ready(development_split())`` is ``False``
    because there is no held-out validation or test landscape.
    """
    return LandscapeSplit(
        split_name="development",
        train_landscape_ids=(SMALL_VECTOR_001_ID,),
        validation_landscape_ids=(),
        test_landscape_ids=(),
        notes=(
            "Single-landscape development split for Stage 4 smoke runs. "
            "Not a transfer-learning split; Stage 5 will introduce real "
            "train/validation/test landscape sets when additional data is "
            "available."
        ),
    )
