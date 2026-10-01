import pytest

from app.pipeline.normalize import parse_date, parse_number


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1 394,67", "1394.67"),
        ("5640,17", "5640.17"),
        ("$12.00", "12.00"),
        ("1,234.56", "1234.56"),
        ("1.234,56", "1234.56"),
        ("10%", "10"),
        ("(12.00)", "-12.00"),
        ("3,00", "3.00"),
        ("1,000", "1000"),
    ],
)
def test_parse_number(raw, expected):
    from decimal import Decimal

    assert parse_number(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["each", "", "abc1", "$"])
def test_parse_number_rejects(raw):
    assert parse_number(raw) is None


@pytest.mark.parametrize(
    "raw,order,iso",
    [
        ("04/13/2013", "MDY", "2013-04-13"),
        ("13/04/2013", "MDY", "2013-04-13"),
        ("04/05/2020", "MDY", "2020-04-05"),
        ("04/05/2020", "DMY", "2020-05-04"),
        ("Mar 13, 2022", "MDY", "2022-03-13"),
        ("2022-03-13", "MDY", "2022-03-13"),
        ("garbage", "MDY", None),
        ("31/02/2020", "MDY", None),
        (None, "MDY", None),
    ],
)
def test_parse_date(raw, order, iso):
    assert parse_date(raw, order) == iso
