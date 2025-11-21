from __future__ import annotations

import pytest

from phes_assessment.climate import _infer_month


def test_infer_month_from_year_month() -> None:
    assert _infer_month("2023-01", 5) == 1
    assert _infer_month("2023_11", 5) == 11


def test_infer_month_from_single_number_without_year() -> None:
    assert _infer_month("band 07", 2) == 7
    assert _infer_month("month=12", 2) == 12


def test_infer_month_fallback_to_band_index() -> None:
    assert _infer_month("no info", 4) == 4


@pytest.mark.parametrize("desc", ["2023", "2020-20", "99"])
def test_infer_month_rejects_invalid_matches(desc: str) -> None:
    assert _infer_month(desc, 6) == 6
