// Numerical regression checks for the collision broadphase helpers.
//
// This is intentionally a small standalone harness. It does not construct a
// robot model or touch hardware, so it can run offline while the controller is
// stopped. The randomized checks compare the SAT decision with Coal's exact
// signed box-to-box distance.
#include "wbc_rt/inner.hpp"

#include <Eigen/Geometry>

#include <coal/distance.h>
#include <coal/math/transform.h>
#include <coal/shape/geometric_shapes.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

using wbc_rt::collision_broadphase::Aabb;
using wbc_rt::collision_broadphase::Obb;
using wbc_rt::collision_broadphase::from_bounds;
using wbc_rt::collision_broadphase::obb_lower_bound;
using wbc_rt::collision_broadphase::obb_separates_above;
using wbc_rt::collision_broadphase::transformed;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Eigen::Matrix3d rotation(double angle, const Eigen::Vector3d& axis) {
  return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}

Eigen::Matrix3d random_rotation(std::mt19937_64* rng) {
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::Vector3d axis(normal(*rng), normal(*rng), normal(*rng));
  if (axis.squaredNorm() < 1.0e-18) axis = Eigen::Vector3d::UnitX();
  std::uniform_real_distribution<double> angle(-3.14159265358979323846,
                                                3.14159265358979323846);
  return rotation(angle(*rng), axis);
}

Eigen::Vector3d random_vector(std::mt19937_64* rng, double low, double high) {
  std::uniform_real_distribution<double> sample(low, high);
  return Eigen::Vector3d(sample(*rng), sample(*rng), sample(*rng));
}

double exact_box_distance(const coal::Box& a, const Eigen::Matrix3d& ra,
                          const Eigen::Vector3d& ta, const coal::Box& b,
                          const Eigen::Matrix3d& rb,
                          const Eigen::Vector3d& tb) {
  const coal::Transform3s tf_a(ra, ta);
  const coal::Transform3s tf_b(rb, tb);
  const coal::DistanceRequest request(true, true);
  coal::DistanceResult result;
  return static_cast<double>(
      coal::distance(&a, tf_a, &b, tf_b, request, result));
}

void test_rotated_box_encloses_local_corners() {
  const Aabb local =
      from_bounds(Eigen::Vector3d(-0.04, -0.02, -0.03),
                  Eigen::Vector3d(0.04, 0.02, 0.03));
  require(local.valid, "finite positive local bounds must be valid");

  const Eigen::Matrix3d r =
      rotation(0.71, Eigen::Vector3d(1.0, 2.0, -0.5));
  const Eigen::Vector3d t(0.13, -0.07, 0.21);
  const Obb world = transformed(local, r, t);
  require(world.valid, "a proper rotation must produce a valid OBB");
  require((world.center - t).norm() < 1.0e-14,
          "OBB center must follow the placement");
  require((world.rotation - r).cwiseAbs().maxCoeff() < 1.0e-14,
          "OBB rotation must be cached");
  require((world.halfwidth - local.halfwidth).cwiseAbs().maxCoeff() < 1.0e-14,
          "OBB half-widths must remain in local axes");

  for (int sx : {-1, 1}) {
    for (int sy : {-1, 1}) {
      for (int sz : {-1, 1}) {
        const Eigen::Vector3d local_corner(
            sx * local.halfwidth.x(), sy * local.halfwidth.y(),
            sz * local.halfwidth.z());
        const Eigen::Vector3d world_corner = t + r * local_corner;
        const Eigen::Vector3d coordinates =
            world.rotation.transpose() * (world_corner - world.center);
        require((coordinates.cwiseAbs() - world.halfwidth).maxCoeff() < 1.0e-14,
                "transformed OBB must enclose every local corner");
      }
    }
  }
}

void test_separation_and_threshold_epsilon() {
  const Aabb local_a =
      from_bounds(Eigen::Vector3d(-0.035, -0.012, -0.018),
                  Eigen::Vector3d(0.035, 0.012, 0.018));
  const Aabb local_b =
      from_bounds(Eigen::Vector3d(-0.022, -0.016, -0.011),
                  Eigen::Vector3d(0.022, 0.016, 0.011));
  const Obb a = transformed(local_a, Eigen::Matrix3d::Identity(),
                             Eigen::Vector3d::Zero());
  const Obb b = transformed(
      local_b, rotation(0.83, Eigen::Vector3d(-0.4, 0.7, 0.2)),
      Eigen::Vector3d(0.20, 0.01, -0.02));
  require(a.valid && b.valid, "test boxes must be valid");

  const double threshold = 0.05;
  require(obb_separates_above(a, b, threshold),
          "a box gap above threshold must be rejected");
  require(obb_lower_bound(a, b) > threshold,
          "SAT lower bound must expose the separating gap");

  // The explicit one-nanometre margin prevents a numerically exact threshold
  // contact from being filtered.
  const double boundary = obb_lower_bound(a, b) - 1.0e-9;
  require(!obb_separates_above(a, b, boundary),
          "threshold margin must retain a boundary pair");
}

void test_active_rotated_pairs_are_never_filtered() {
  const Aabb local_a =
      from_bounds(Eigen::Vector3d(-0.040, -0.020, -0.030),
                  Eigen::Vector3d(0.040, 0.020, 0.030));
  const Aabb local_b =
      from_bounds(Eigen::Vector3d(-0.025, -0.030, -0.015),
                  Eigen::Vector3d(0.025, 0.030, 0.015));
  const double threshold = 0.20;
  const Eigen::Vector3d center_delta(0.080, 0.030, 0.020);
  const double exact_upper_bound = center_delta.norm();
  require(exact_upper_bound < threshold,
          "fixture pair must be inside the activation threshold");

  const double angles[] = {0.0, 0.21, 0.73, 1.47, 2.61, -0.64};
  for (double angle : angles) {
    const Obb a = transformed(
        local_a, rotation(angle, Eigen::Vector3d(1.0, 0.0, 0.2)),
        Eigen::Vector3d::Zero());
    const Obb b = transformed(
        local_b, rotation(-0.7 * angle, Eigen::Vector3d(0.3, 1.0, -0.4)),
        center_delta);
    require(a.valid && b.valid, "rotated active fixture must be valid");
    require(obb_lower_bound(a, b) <= exact_upper_bound + 1.0e-12,
            "OBB lower bound cannot exceed an exact-distance upper bound");
    require(!obb_separates_above(a, b, threshold),
            "an exact-active pair must survive every tested pose");
  }
}

void test_random_sat_filter_matches_coal_distance() {
  const Eigen::Vector3d half_a(0.120, 0.035, 0.025);
  const Eigen::Vector3d half_b(0.090, 0.050, 0.020);
  const Aabb local_a = from_bounds(-half_a, half_a);
  const Aabb local_b = from_bounds(-half_b, half_b);
  coal::Box shape_a(2.0 * half_a.x(), 2.0 * half_a.y(), 2.0 * half_a.z());
  coal::Box shape_b(2.0 * half_b.x(), 2.0 * half_b.y(), 2.0 * half_b.z());
  shape_a.computeLocalAABB();
  shape_b.computeLocalAABB();

  constexpr double threshold = 0.05;
  constexpr double exact_margin = 1.0e-10;
  std::mt19937_64 rng(0x4f42425f736f756eULL);
  int filtered = 0;
  int near_threshold = 0;
  int overlapping = 0;

  auto check_case = [&](const char* label, const Eigen::Matrix3d& ra,
                        const Eigen::Vector3d& ta, const Eigen::Matrix3d& rb,
                        const Eigen::Vector3d& tb) {
    const Obb a = transformed(local_a, ra, ta);
    const Obb b = transformed(local_b, rb, tb);
    require(a.valid && b.valid, "random proper poses must produce valid OBBs");
    const double exact =
        exact_box_distance(shape_a, ra, ta, shape_b, rb, tb);
    require(std::isfinite(exact), "Coal must return a finite box distance");
    const bool reject = obb_separates_above(a, b, threshold);
    if (exact >= threshold - 1.0e-5 && exact <= threshold + 1.0e-5) {
      ++near_threshold;
    }
    if (exact < -1.0e-6) ++overlapping;
    if (reject) {
      ++filtered;
      if (!(exact > threshold + exact_margin)) {
        std::cerr << "SAT rejected " << label << " with exact=" << exact
                  << " threshold=" << threshold << '\n';
        throw std::runtime_error(
            "OBB filter rejected a pair whose Coal distance is not above "
            "the activation threshold");
      }
    }
    if (exact <= threshold && reject) {
      throw std::runtime_error(
          "OBB filter rejected an exact-active or threshold pair");
    }
  };

  // Explicit parallel and nearly-parallel poses exercise the cross-axis
  // degeneracy path as well as the one-nanometre threshold margin.
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d almost_parallel =
      rotation(1.0e-13, Eigen::Vector3d::UnitZ());
  const Eigen::Matrix3d near_parallel =
      rotation(1.0e-10, Eigen::Vector3d::UnitZ());
  check_case("parallel-overlap", identity, Eigen::Vector3d::Zero(), identity,
             Eigen::Vector3d(0.03, 0.0, 0.0));
  const double x_at_threshold = half_a.x() + half_b.x() + threshold;
  check_case("parallel-threshold", identity, Eigen::Vector3d::Zero(), identity,
             Eigen::Vector3d(x_at_threshold, 0.0, 0.0));
  check_case("parallel-just-separated", identity, Eigen::Vector3d::Zero(),
             identity, Eigen::Vector3d(x_at_threshold + 2.0e-9, 0.0, 0.0));
  check_case("near-parallel-separated", identity, Eigen::Vector3d::Zero(),
             near_parallel,
             Eigen::Vector3d(x_at_threshold + 2.0e-4, 0.0, 0.0));
  check_case("cross-axis-degenerate", identity, Eigen::Vector3d::Zero(),
             almost_parallel,
             Eigen::Vector3d(x_at_threshold + 2.0e-4, 0.0, 0.0));

  // Hundreds of unrestricted poses cover arbitrary face/cross-axis choices.
  for (int i = 0; i < 360; ++i) {
    check_case("random", random_rotation(&rng), random_vector(&rng, -0.25, 0.25),
               random_rotation(&rng), random_vector(&rng, -0.45, 0.45));
  }

  // Same-center boxes are guaranteed to overlap, independent of orientation.
  for (int i = 0; i < 40; ++i) {
    const Eigen::Vector3d center = random_vector(&rng, -0.20, 0.20);
    check_case("overlap", random_rotation(&rng), center, random_rotation(&rng),
               center);
  }

  // Bisection along a random direction creates exact distances at the
  // activation threshold for arbitrary relative rotations. This makes the
  // test sensitive to an accidental strict/weak comparison or epsilon loss.
  for (int i = 0; i < 80; ++i) {
    const Eigen::Matrix3d ra = random_rotation(&rng);
    const Eigen::Matrix3d rb = random_rotation(&rng);
    const Eigen::Vector3d ta = random_vector(&rng, -0.10, 0.10);
    Eigen::Vector3d direction = random_vector(&rng, -1.0, 1.0);
    if (direction.squaredNorm() < 1.0e-12) direction = Eigen::Vector3d::UnitX();
    direction.normalize();
    double lo = 0.0;
    double hi = 1.0;
    for (int iteration = 0; iteration < 50; ++iteration) {
      const double mid = 0.5 * (lo + hi);
      const Eigen::Vector3d tb = ta + direction * mid;
      const double exact =
          exact_box_distance(shape_a, ra, ta, shape_b, rb, tb);
      if (exact > threshold) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    check_case("random-threshold", ra, ta, rb, ta + direction * hi);
  }

  require(filtered > 0, "random fixture must exercise SAT rejections");
  require(near_threshold >= 20,
          "random fixture must contain many threshold-near poses");
  require(overlapping >= 40,
          "overlap fixture must remain exact-active and unfiltered");
}

void test_invalid_bounds_fall_back_to_sphere() {
  const Aabb valid =
      from_bounds(Eigen::Vector3d(-0.01, -0.01, -0.01),
                  Eigen::Vector3d(0.01, 0.01, 0.01));
  const Aabb degenerate =
      from_bounds(Eigen::Vector3d(0.0, -0.01, -0.01),
                  Eigen::Vector3d(0.0, 0.01, 0.01));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Aabb nonfinite =
      from_bounds(Eigen::Vector3d(-nan, -0.01, -0.01),
                  Eigen::Vector3d(0.01, 0.01, 0.01));
  require(!degenerate.valid, "degenerate bounds must be unavailable");
  require(!nonfinite.valid, "nonfinite bounds must be unavailable");

  const Obb good = transformed(valid, Eigen::Matrix3d::Identity(),
                               Eigen::Vector3d::Zero());
  const Obb bad = transformed(degenerate, Eigen::Matrix3d::Identity(),
                              Eigen::Vector3d(1.0, 0.0, 0.0));
  require(good.valid && !bad.valid, "invalid local bounds must invalidate OBB");
  require(!obb_separates_above(bad, good, 0.01),
          "invalid OBBs must never suppress an exact query");
  require(!std::isfinite(obb_lower_bound(bad, good)),
          "invalid OBB lower bound must signal sphere fallback");

  const Eigen::Matrix3d nonrigid =
      (Eigen::Vector3d(1.0, 1.0, 1.1).asDiagonal());
  const Obb malformed = transformed(valid, nonrigid, Eigen::Vector3d::Zero());
  require(!malformed.valid,
          "a non-rigid placement must disable the OBB filter");
}

}  // namespace

int main() {
  try {
    test_rotated_box_encloses_local_corners();
    test_separation_and_threshold_epsilon();
    test_active_rotated_pairs_are_never_filtered();
    test_random_sat_filter_matches_coal_distance();
    test_invalid_bounds_fall_back_to_sphere();
  } catch (const std::exception& exc) {
    std::cerr << "collision broadphase regression failed: " << exc.what()
              << '\n';
    return 1;
  }
  std::cout << "collision broadphase regression passed\n";
  return 0;
}
