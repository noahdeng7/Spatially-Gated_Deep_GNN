"""
Unit tests for the distance functions.

Reference values come from three independent places, on purpose:

  1. Closed-form geodetic constants (quarter meridian, one degree of longitude
     at the equator) that can be checked against a geodesy reference and do not
     depend on any library.
  2. geopy.distance.geodesic, an independent implementation of Karney's
     algorithm -- if pyproj and geopy agree to sub-millimetre, a wiring error in
     our wrapper is essentially ruled out.
  3. Hand-checked city pairs at loose tolerance, which catch gross errors like a
     lat/lon transposition that the tight tests might not.

The transposition tests matter most. Swapping lat and lon does not raise -- it
returns a plausible-looking number for the wrong pair of points, which is the
kind of bug that survives all the way into a published table.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# --- geodetic constants -----------------------------------------------------
# Quarter meridian on WGS84: distance from equator to pole along a meridian.
QUARTER_MERIDIAN_KM = 10001.965729
# One degree of longitude along the equator on WGS84 (= a/  ... 2*pi*a/360).
ONE_DEG_LON_EQUATOR_KM = 111.319491
# One degree of latitude at the equator on WGS84 (shorter than a degree of
# longitude, because the ellipsoid is flattened at the poles).
ONE_DEG_LAT_EQUATOR_KM = 110.574389

# Reference pairs (decimal degrees) with expected geodesic distance.
#
# PROVENANCE, because it matters for what these tests are worth: the expected
# values were cross-checked against geopy.distance.geodesic, an independent
# Karney implementation, and each was separately sanity-checked by hand against
# a flat-earth approximation (delta-lon scaled by cos(mean lat), delta-lat at
# ~111 km/degree, combined in quadrature) which agrees to within a percent or so
# at these distances.
#
# They are NOT independent of pyproj in the strong sense. That job belongs to
# TestGeodesicKnownValues, whose expected values are closed-form geodetic
# constants that hold regardless of implementation. These pairs exist to catch
# gross errors -- a lat/lon transposition, a metres/kilometres slip -- at
# realistic Mexican scales, which is why the tolerances are loose.
CITY_PAIRS = [
    # (name_a, lat_a, lon_a, name_b, lat_b, lon_b, approx_km, tol_km)
    ("Mexico City", 19.4326, -99.1332, "Guadalajara", 20.6597, -103.3496, 461.5, 15.0),
    ("Mexico City", 19.4326, -99.1332, "Monterrey", 25.6866, -100.3161, 703.2, 20.0),
    ("Tijuana", 32.5149, -117.0382, "Cancun", 21.1619, -86.8515, 3238.7, 60.0),
    ("Merida", 20.9674, -89.5926, "Oaxaca", 17.0732, -96.7266, 865.8, 25.0),
]


class TestGeodesicKnownValues:
    def test_one_degree_longitude_at_equator(self, distance_mod):
        d = float(distance_mod.geodesic_km(0.0, 0.0, 0.0, 1.0))
        assert d == pytest.approx(ONE_DEG_LON_EQUATOR_KM, abs=0.001)

    def test_one_degree_latitude_at_equator(self, distance_mod):
        d = float(distance_mod.geodesic_km(0.0, 0.0, 1.0, 0.0))
        assert d == pytest.approx(ONE_DEG_LAT_EQUATOR_KM, abs=0.001)

    def test_equator_to_pole_is_quarter_meridian(self, distance_mod):
        d = float(distance_mod.geodesic_km(0.0, 0.0, 90.0, 0.0))
        assert d == pytest.approx(QUARTER_MERIDIAN_KM, abs=0.01)

    def test_a_degree_of_latitude_exceeds_a_degree_of_longitude_at_60N(self, distance_mod):
        """
        At 60N a degree of longitude spans roughly half what it does at the
        equator (cos 60 = 0.5), while a degree of latitude is barely changed.
        Getting this backwards is the signature of a transposed argument.
        """
        lon_deg = float(distance_mod.geodesic_km(60.0, 0.0, 60.0, 1.0))
        lat_deg = float(distance_mod.geodesic_km(60.0, 0.0, 61.0, 0.0))
        assert lon_deg == pytest.approx(55.8, abs=0.5)
        assert lat_deg == pytest.approx(111.4, abs=0.5)
        assert lat_deg > lon_deg

    def test_zero_distance_for_identical_points(self, distance_mod):
        assert float(distance_mod.geodesic_km(19.4326, -99.1332, 19.4326, -99.1332)) == \
            pytest.approx(0.0, abs=1e-9)


class TestGeodesicAgainstIndependentImplementation:
    """Cross-check pyproj against geopy -- two separate Karney implementations."""

    @pytest.mark.parametrize("pair", CITY_PAIRS, ids=lambda p: f"{p[0]}-{p[3]}")
    def test_matches_geopy(self, distance_mod, pair):
        geopy_distance = pytest.importorskip("geopy.distance")
        _, lat_a, lon_a, _, lat_b, lon_b, _, _ = pair
        ours = float(distance_mod.geodesic_km(lat_a, lon_a, lat_b, lon_b))
        theirs = geopy_distance.geodesic((lat_a, lon_a), (lat_b, lon_b)).km
        # Sub-millimetre agreement. Anything worse means a wiring error.
        assert ours == pytest.approx(theirs, abs=1e-6)


class TestGeodesicHandChecked:
    @pytest.mark.parametrize("pair", CITY_PAIRS, ids=lambda p: f"{p[0]}-{p[3]}")
    def test_city_pair_within_tolerance(self, distance_mod, pair):
        name_a, lat_a, lon_a, name_b, lat_b, lon_b, approx, tol = pair
        d = float(distance_mod.geodesic_km(lat_a, lon_a, lat_b, lon_b))
        assert d == pytest.approx(approx, abs=tol), (
            f"{name_a}-{name_b}: got {d:.1f} km, expected ~{approx} +/- {tol}"
        )

    def test_transposing_lat_and_lon_gives_a_different_answer(self, distance_mod):
        """
        Guards the (lon, lat) vs (lat, lon) argument order into pyproj.Geod.inv.

        This test is here because the bug it catches is silent: a transposition
        returns a perfectly plausible distance for entirely the wrong points.
        For Mexican coordinates the transposed pair lands off Somalia, so the
        answer is wildly different -- but only if you check.
        """
        correct = float(distance_mod.geodesic_km(19.4326, -99.1332, 20.6597, -103.3496))
        transposed = float(distance_mod.geodesic_km(-99.1332, 19.4326, -103.3496, 20.6597))
        assert not math.isclose(correct, transposed, rel_tol=0.01)
        assert correct == pytest.approx(461.5, abs=15.0)


class TestGeodesicProperties:
    def test_symmetric(self, distance_mod):
        ab = float(distance_mod.geodesic_km(19.4326, -99.1332, 25.6866, -100.3161))
        ba = float(distance_mod.geodesic_km(25.6866, -100.3161, 19.4326, -99.1332))
        assert ab == pytest.approx(ba, abs=1e-9)

    def test_triangle_inequality(self, distance_mod):
        a = (19.4326, -99.1332)
        b = (20.6597, -103.3496)
        c = (25.6866, -100.3161)
        ab = float(distance_mod.geodesic_km(*a, *b))
        bc = float(distance_mod.geodesic_km(*b, *c))
        ac = float(distance_mod.geodesic_km(*a, *c))
        assert ac <= ab + bc + 1e-9

    def test_vectorised_matches_scalar(self, distance_mod):
        lat1 = np.array([19.4326, 25.6866, 32.5149])
        lon1 = np.array([-99.1332, -100.3161, -117.0382])
        lat2 = np.array([20.6597, 21.1619, 20.9674])
        lon2 = np.array([-103.3496, -86.8515, -89.5926])
        vec = distance_mod.geodesic_km(lat1, lon1, lat2, lon2)
        for i in range(3):
            scalar = float(distance_mod.geodesic_km(lat1[i], lon1[i], lat2[i], lon2[i]))
            assert float(vec[i]) == pytest.approx(scalar, abs=1e-9)

    def test_antipodal_points_do_not_break(self, distance_mod):
        """
        Antipodes are where the spherical law of cosines loses all precision and
        naive haversine degrades. Karney's algorithm handles them; this asserts
        we get roughly half a circumference rather than a NaN.
        """
        d = float(distance_mod.geodesic_km(0.0, 0.0, 0.0, 180.0))
        assert np.isfinite(d)
        assert d == pytest.approx(20003.93, abs=1.0)


class TestGreatCircle:
    def test_close_to_geodesic_at_mexican_scales(self, distance_mod):
        """
        Spherical and ellipsoidal distances differ by a few tenths of a percent
        at these latitudes. Both are retained in the panel; this pins the size
        of the gap so a change in it would be noticed.
        """
        geo = float(distance_mod.geodesic_km(19.4326, -99.1332, 25.6866, -100.3161))
        gc = float(distance_mod.great_circle_km(19.4326, -99.1332, 25.6866, -100.3161))
        rel = abs(gc - geo) / geo
        assert rel < 0.005, f"great-circle differs from geodesic by {rel:.4%}"
        assert rel > 0, "identical values suggest great_circle_km is not spherical"

    def test_one_degree_longitude_at_equator_on_a_sphere(self, distance_mod):
        d = float(distance_mod.great_circle_km(0.0, 0.0, 0.0, 1.0))
        # 2*pi*6371.0088 / 360
        assert d == pytest.approx(111.19493, abs=0.001)

    def test_symmetric(self, distance_mod):
        ab = float(distance_mod.great_circle_km(19.4, -99.1, 25.7, -100.3))
        ba = float(distance_mod.great_circle_km(25.7, -100.3, 19.4, -99.1))
        assert ab == pytest.approx(ba, abs=1e-9)


class TestInternalDistance:
    def test_head_mayer_formula_on_a_known_disc(self, distance_mod):
        """
        For a disc of radius 10 km, area = pi * 100, so sqrt(area/pi) = 10 and
        the Head-Mayer internal distance is 0.67 * 10 = 6.7 km.
        """
        d = float(distance_mod.internal_distance_km(math.pi * 100.0, 0.67))
        assert d == pytest.approx(6.7, abs=1e-9)

    def test_scales_with_square_root_of_area(self, distance_mod):
        small = float(distance_mod.internal_distance_km(100.0))
        large = float(distance_mod.internal_distance_km(400.0))
        assert large == pytest.approx(2.0 * small, abs=1e-9)

    def test_zero_area_gives_zero(self, distance_mod):
        assert float(distance_mod.internal_distance_km(0.0)) == pytest.approx(0.0)

    def test_negative_area_raises(self, distance_mod):
        with pytest.raises(ValueError):
            distance_mod.internal_distance_km(-1.0)

    def test_coefficient_is_applied(self, distance_mod):
        default = float(distance_mod.internal_distance_km(1000.0))
        doubled = float(distance_mod.internal_distance_km(1000.0, coefficient=1.34))
        assert doubled == pytest.approx(2.0 * default, abs=1e-9)

    def test_realistic_municipality_magnitude(self, distance_mod):
        """
        A 1,000 km2 municipality (roughly the Mexican median order of magnitude)
        should get an internal distance of ~12 km -- the same order as the
        distance to its neighbours, which is the point of the correction.
        """
        d = float(distance_mod.internal_distance_km(1000.0))
        assert 10.0 < d < 15.0

    def test_vectorised(self, distance_mod):
        areas = np.array([100.0, 400.0, 900.0])
        out = distance_mod.internal_distance_km(areas)
        assert out.shape == (3,)
        assert float(out[1]) == pytest.approx(2.0 * float(out[0]), abs=1e-9)
