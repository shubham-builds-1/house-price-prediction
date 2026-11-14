"""Tests for the live-demo model in house_price/market.py (9 cities, 121 localities)."""

import pytest

from house_price import market
from house_price.market import (
    FURNISHING,
    LOCALITY_CITY,
    LOCALITY_RATE,
    RATES,
    format_price,
    predict_price,
    simulate_listings,
    train,
)

BASE = dict(area=1200, bedrooms=3, bathrooms=2, parking=1, furnishing="Semi-furnished", age=5)


@pytest.fixture(scope="module")
def trained():
    model, r2 = train(save=False)  # do not overwrite models/market_model.joblib during tests
    return model, r2


# ---- coverage of the rate table -------------------------------------------------------------

def test_covers_nine_cities_and_121_localities():
    assert len(RATES) == 9
    assert sum(len(v) for v in RATES.values()) == 121


def test_locality_lookups_are_consistent():
    # a locality name must not appear in two cities, otherwise the flat lookups would collide
    assert len(LOCALITY_RATE) == 121
    assert len(LOCALITY_CITY) == 121
    for city, localities in RATES.items():
        for name, rate in localities.items():
            assert LOCALITY_CITY[name] == city
            assert LOCALITY_RATE[name] == rate


def test_all_rates_are_plausible_rupees_per_sqft():
    rates = [r for d in RATES.values() for r in d.values()]
    assert min(rates) > 1_000 and max(rates) < 200_000


# ---- price adjustment factors ---------------------------------------------------------------

def test_older_houses_get_lower_factor():
    new = float(market._price_factor(1000, 2, 2, 1, "Semi-furnished", 0)[0])
    old = float(market._price_factor(1000, 2, 2, 1, "Semi-furnished", 30)[0])
    assert old < new


def test_furnished_and_parking_raise_factor():
    unf = float(market._price_factor(1000, 2, 2, 0, "Unfurnished", 5)[0])
    fur = float(market._price_factor(1000, 2, 2, 2, "Furnished", 5)[0])
    assert fur > unf


# ---- simulated training data ----------------------------------------------------------------

def test_simulated_listings_are_valid_and_reproducible():
    a = simulate_listings(n=300, seed=7)
    b = simulate_listings(n=300, seed=7)
    assert a.equals(b)
    assert list(a.columns) == market.FEATURES + ["price"]
    assert (a["price"] > 0).all()
    assert set(a["locality"]) <= set(LOCALITY_CITY)
    assert set(a["furnishing"]) <= set(FURNISHING)


# ---- trained model --------------------------------------------------------------------------

def test_model_fits_simulated_holdout(trained):
    _, r2 = trained
    assert r2 > 0.9  # only says the model learned the simulated pattern, not real-world accuracy


def test_prediction_output_shape(trained):
    model, _ = trained
    out = predict_price(model, "Pune", "Baner", **BASE)
    assert set(out) == {"price", "per_sqft", "benchmark_per_sqft"}
    assert out["price"] > 0
    assert out["per_sqft"] == pytest.approx(out["price"] / BASE["area"])
    assert out["benchmark_per_sqft"] == RATES["Pune"]["Baner"]


def test_unknown_locality_raises(trained):
    model, _ = trained
    with pytest.raises(ValueError):
        predict_price(model, "Pune", "Atlantis", **BASE)
    with pytest.raises(ValueError):  # real locality, wrong city
        predict_price(model, "Pune", "Bandra West", **BASE)


@pytest.mark.parametrize(
    "city,locality",
    [("Delhi", "Dwarka Sector 26"), ("Mumbai", "Thane (Majiwada)"), ("Bengaluru", "Koramangala"),
     ("Hyderabad", "Gachibowli"), ("Kolkata", "New Town"), ("Noida", "Sector 150")],
)
def test_estimate_stays_near_locality_benchmark(trained, city, locality):
    model, _ = trained
    out = predict_price(model, city, locality, **BASE)
    assert 0.8 < out["per_sqft"] / out["benchmark_per_sqft"] < 1.2


def test_expensive_locality_costs_more_than_cheap_one(trained):
    model, _ = trained
    pricey = predict_price(model, "Mumbai", "Bandra West", **BASE)["price"]
    cheap = predict_price(model, "Mumbai", "Thane (Majiwada)", **BASE)["price"]
    assert pricey > cheap


def test_bigger_and_newer_costs_more(trained):
    model, _ = trained
    small = predict_price(model, "Bengaluru", "Koramangala", **{**BASE, "area": 800})["price"]
    big = predict_price(model, "Bengaluru", "Koramangala", **{**BASE, "area": 2400})["price"]
    assert big > small
    old = predict_price(model, "Bengaluru", "Koramangala", **{**BASE, "age": 30})["price"]
    new = predict_price(model, "Bengaluru", "Koramangala", **{**BASE, "age": 0})["price"]
    assert new > old


# ---- formatting -----------------------------------------------------------------------------

def test_format_price():
    assert format_price(8_500_000) == "₹85.0 lakh"
    assert format_price(25_000_000) == "₹2.50 crore"
    assert format_price(10_000_000) == "₹1.00 crore"
