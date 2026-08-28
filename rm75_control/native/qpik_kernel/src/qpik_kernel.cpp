#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

using InArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMap = Eigen::Map<RowMat>;
using ConstRowMap = Eigen::Map<const RowMat>;
using VecMap = Eigen::Map<Eigen::VectorXd>;
using ConstVecMap = Eigen::Map<const Eigen::VectorXd>;

py::array_t<double> as_mat(const std::vector<double>& buf, ssize_t r, ssize_t c) {
  py::array_t<double> out({r, c});
  if (r > 0 && c > 0) {
    std::memcpy(
        out.mutable_data(),
        buf.data(),
        sizeof(double) * static_cast<size_t>(r * c)
    );
  }
  return out;
}

std::vector<int> load_ints(const py::array& a) {
  const auto buf = a.request();
  const ssize_t n = buf.size;
  std::vector<int> out(static_cast<size_t>(std::max<ssize_t>(n, 0)));
  if (n <= 0) {
    return out;
  }
  if (buf.itemsize == static_cast<ssize_t>(sizeof(std::int64_t))) {
    const auto* p = static_cast<const std::int64_t*>(buf.ptr);
    for (ssize_t i = 0; i < n; ++i) {
      out[static_cast<size_t>(i)] = static_cast<int>(p[i]);
    }
  } else if (buf.itemsize == static_cast<ssize_t>(sizeof(std::int32_t))) {
    const auto* p = static_cast<const std::int32_t*>(buf.ptr);
    for (ssize_t i = 0; i < n; ++i) {
      out[static_cast<size_t>(i)] = static_cast<int>(p[i]);
    }
  } else {
    throw std::runtime_error("index array must be int32 or int64");
  }
  return out;
}

int rows2(const InArr& a) {
  if (a.ndim() == 0) {
    return 0;
  }
  return static_cast<int>(a.shape(0));
}

int cols2(const InArr& a, int fallback) {
  if (a.ndim() < 2) {
    return fallback;
  }
  return static_cast<int>(a.shape(1));
}

py::tuple build_wbc_inequalities(
    int nv,
    int n_task_slack,
    InArr lo_box,
    InArr hi_box,
    InArr cbf_jac,
    InArr cbf_lower,
    py::array cbf_slots,
    int max_cbf_rows,
    int n_pref_slack,
    int max_pref_rows,
    InArr pref_jac,
    py::array pref_slack_col,
    InArr pref_lower) {
  if (nv <= 0 || n_task_slack < 0 || max_cbf_rows < 0 || n_pref_slack < 0 ||
      max_pref_rows < 0) {
    throw std::runtime_error("invalid dimensions");
  }
  const int n_in = nv + max_cbf_rows + max_pref_rows + n_pref_slack;
  const int n_var = nv + n_task_slack + n_pref_slack;
  std::vector<double> Cbuf(static_cast<size_t>(n_in) * static_cast<size_t>(n_var), 0.0);
  std::vector<double> lbuf(static_cast<size_t>(n_in), -INFINITY);
  std::vector<double> ubuf(static_cast<size_t>(n_in), INFINITY);
  auto C_at = [&](int r, int c) -> double& {
    return Cbuf[static_cast<size_t>(r) * static_cast<size_t>(n_var) + static_cast<size_t>(c)];
  };

  const int n_lo = static_cast<int>(lo_box.size());
  const int n_hi = static_cast<int>(hi_box.size());
  const double* lo_p = lo_box.data();
  const double* hi_p = hi_box.data();
  for (int i = 0; i < nv; ++i) {
    C_at(i, i) = 1.0;
    if (i < n_lo) {
      lbuf[static_cast<size_t>(i)] = lo_p[i];
    }
    if (i < n_hi) {
      ubuf[static_cast<size_t>(i)] = hi_p[i];
    }
  }

  const int n_cbf = rows2(cbf_jac);
  const int cbf_cols = cols2(cbf_jac, nv);
  const auto slots = load_ints(cbf_slots);
  const double* jac_p = cbf_jac.data();
  const double* cbf_lo_p = cbf_lower.data();
  const int n_cbf_lo = static_cast<int>(cbf_lower.size());
  if (!slots.empty() && static_cast<int>(slots.size()) == n_cbf && n_cbf > 0) {
    for (int k = 0; k < n_cbf; ++k) {
      const int i = slots[static_cast<size_t>(k)];
      if (i < 0 || i >= max_cbf_rows) {
        continue;
      }
      for (int j = 0; j < nv && j < cbf_cols; ++j) {
        C_at(nv + i, j) = jac_p[k * cbf_cols + j];
      }
      if (k < n_cbf_lo) {
        lbuf[static_cast<size_t>(nv + i)] = cbf_lo_p[k];
      }
    }
  } else {
    const int n_use = std::min(n_cbf, max_cbf_rows);
    for (int i = 0; i < n_use; ++i) {
      for (int j = 0; j < nv && j < cbf_cols; ++j) {
        C_at(nv + i, j) = jac_p[i * cbf_cols + j];
      }
      if (i < n_cbf_lo) {
        lbuf[static_cast<size_t>(nv + i)] = cbf_lo_p[i];
      }
    }
  }

  const int pref_base = nv + max_cbf_rows;
  const int n_pref_in = rows2(pref_jac);
  const int pref_cols = cols2(pref_jac, nv);
  const int n_pref = std::min(n_pref_in, max_pref_rows);
  const auto pref_slots = load_ints(pref_slack_col);
  const double* pref_j = pref_jac.data();
  const double* pref_l = pref_lower.data();
  const int n_pref_lo = static_cast<int>(pref_lower.size());
  for (int k = 0; k < n_pref; ++k) {
    for (int j = 0; j < nv && j < pref_cols; ++j) {
      C_at(pref_base + k, j) = pref_j[k * pref_cols + j];
    }
    if (k < static_cast<int>(pref_slots.size())) {
      const int s_idx = pref_slots[static_cast<size_t>(k)];
      if (s_idx >= 0 && s_idx < n_pref_slack) {
        C_at(pref_base + k, nv + n_task_slack + s_idx) = 1.0;
      }
    }
    if (k < n_pref_lo) {
      lbuf[static_cast<size_t>(pref_base + k)] = pref_l[k];
    }
  }

  const int slack_base = pref_base + max_pref_rows;
  for (int k = 0; k < n_pref_slack; ++k) {
    C_at(slack_base + k, nv + n_task_slack + k) = 1.0;
    lbuf[static_cast<size_t>(slack_base + k)] = 0.0;
  }
  // Return l/u as STL vectors: pybind11/numpy drops ±inf on a 1-D array_t.
  return py::make_tuple(as_mat(Cbuf, n_in, n_var), lbuf, ubuf);
}

std::vector<double> singular_values(InArr J) {
  if (J.ndim() != 2) {
    throw std::runtime_error("J must be 2-D");
  }
  const int m = static_cast<int>(J.shape(0));
  const int n = static_cast<int>(J.shape(1));
  ConstRowMap Jm(J.data(), m, n);
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jm, Eigen::ComputeThinU);
  std::vector<double> sig(static_cast<size_t>(m), 0.0);
  const int nsig = std::min(m, static_cast<int>(svd.singularValues().size()));
  for (int i = 0; i < nsig; ++i) {
    sig[static_cast<size_t>(i)] = svd.singularValues()(i);
  }
  return sig;
}

std::vector<double> project_nullspace(
    InArr J,
    InArr qdot0,
    double damping,
    InArr M,
    bool use_dyn,
    double m_floor) {
  if (J.ndim() != 2) {
    throw std::runtime_error("J must be 2-D");
  }
  const int m = static_cast<int>(J.shape(0));
  const int nv = static_cast<int>(J.shape(1));
  if (static_cast<int>(qdot0.size()) != nv) {
    throw std::runtime_error("qdot0 size must match J columns");
  }
  ConstRowMap Jm(J.data(), m, nv);
  ConstVecMap q0(qdot0.data(), nv);
  const double lam2 = damping * damping;
  Eigen::MatrixXd N = Eigen::MatrixXd::Identity(nv, nv);
  if (use_dyn && M.ndim() == 2 && static_cast<int>(M.shape(0)) == nv &&
      static_cast<int>(M.shape(1)) == nv) {
    Eigen::MatrixXd Mf =
        ConstRowMap(M.data(), nv, nv) +
        m_floor * Eigen::MatrixXd::Identity(nv, nv);
    const Eigen::MatrixXd Minv = Mf.inverse();
    const Eigen::MatrixXd JMinv = Jm * Minv;
    Eigen::MatrixXd A = JMinv * Jm.transpose();
    A.diagonal().array() += lam2;
    const Eigen::MatrixXd Jbar = Minv * Jm.transpose() * A.ldlt().solve(Eigen::MatrixXd::Identity(m, m));
    N.noalias() -= Jbar * Jm;
  } else {
    Eigen::MatrixXd A = Jm * Jm.transpose();
    A.diagonal().array() += lam2;
    const Eigen::MatrixXd Jd =
        Jm.transpose() * A.ldlt().solve(Eigen::MatrixXd::Identity(m, m));
    N.noalias() -= Jd * Jm;
  }
  const Eigen::VectorXd qn = N * q0;
  return std::vector<double>(qn.data(), qn.data() + qn.size());
}

py::tuple setup_qp1(int nv, int n_task, int n_pref, InArr w_task, InArr J_task) {
  if (nv <= 0 || n_task <= 0 || n_pref < 0) {
    throw std::runtime_error("invalid QP1 dimensions");
  }
  if (w_task.ndim() != 2 || J_task.ndim() != 2) {
    throw std::runtime_error("w_task and J_task must be 2-D");
  }
  if (static_cast<int>(w_task.shape(0)) != n_task ||
      static_cast<int>(w_task.shape(1)) != n_task ||
      static_cast<int>(J_task.shape(0)) != n_task ||
      static_cast<int>(J_task.shape(1)) != nv) {
    throw std::runtime_error("QP1 matrix shape mismatch");
  }
  const int n_var = nv + n_task + n_pref;
  RowMat Hm = RowMat::Zero(n_var, n_var);
  Eigen::VectorXd g = Eigen::VectorXd::Zero(n_var);
  RowMat Am = RowMat::Zero(n_task, n_var);
  Hm.block(nv, nv, n_task, n_task) = ConstRowMap(w_task.data(), n_task, n_task);
  Am.leftCols(nv) = ConstRowMap(J_task.data(), n_task, nv);
  Am.block(0, nv, n_task, n_task) = -Eigen::MatrixXd::Identity(n_task, n_task);
  std::vector<double> Hbuf(static_cast<size_t>(n_var * n_var));
  std::vector<double> Abuf(static_cast<size_t>(n_task * n_var));
  Eigen::Map<RowMat>(Hbuf.data(), n_var, n_var) = Hm;
  Eigen::Map<RowMat>(Abuf.data(), n_task, n_var) = Am;
  return py::make_tuple(
      as_mat(Hbuf, n_var, n_var),
      std::vector<double>(g.data(), g.data() + g.size()),
      as_mat(Abuf, n_task, n_var)
  );
}

py::tuple setup_qp2_costs(
    int nv,
    int n_task,
    int n_pref,
    InArr h_reg,
    InArr qdot_nom,
    InArr slack_w,
    double rail_w,
    double rail_vel,
    InArr smooth,
    InArr qdot_prev) {
  if (nv <= 0 || n_task <= 0 || n_pref < 0) {
    throw std::runtime_error("invalid QP2 dimensions");
  }
  if (static_cast<int>(h_reg.size()) != nv ||
      static_cast<int>(qdot_nom.size()) != nv ||
      static_cast<int>(slack_w.size()) != n_pref ||
      static_cast<int>(smooth.size()) != nv ||
      static_cast<int>(qdot_prev.size()) != nv) {
    throw std::runtime_error("QP2 vector size mismatch");
  }
  const int n_var = nv + n_task + n_pref;
  RowMat Hm = RowMat::Zero(n_var, n_var);
  Eigen::VectorXd gm = Eigen::VectorXd::Zero(n_var);
  ConstVecMap hr(h_reg.data(), nv);
  ConstVecMap qn(qdot_nom.data(), nv);
  ConstVecMap sw(slack_w.data(), n_pref);
  ConstVecMap sm(smooth.data(), nv);
  ConstVecMap qp(qdot_prev.data(), nv);
  Hm.topLeftCorner(nv, nv) = hr.asDiagonal();
  Hm.block(nv, nv, n_task, n_task) =
      1.0e-10 * Eigen::MatrixXd::Identity(n_task, n_task);
  for (int k = 0; k < n_pref; ++k) {
    Hm(nv + n_task + k, nv + n_task + k) = sw(k);
  }
  gm.head(nv) = -hr.cwiseProduct(qn);
  if (rail_w > 0.0) {
    Hm(0, 0) += rail_w;
    gm(0) -= rail_w * rail_vel;
  }
  if (sm.maxCoeff() > 0.0) {
    Hm.topLeftCorner(nv, nv) += sm.asDiagonal();
    gm.head(nv) -= sm.cwiseProduct(qp);
  }
  std::vector<double> Hbuf(static_cast<size_t>(n_var * n_var));
  Eigen::Map<RowMat>(Hbuf.data(), n_var, n_var) = Hm;
  return py::make_tuple(
      as_mat(Hbuf, n_var, n_var),
      std::vector<double>(gm.data(), gm.data() + gm.size())
  );
}

double clip_d(double x, double lo, double hi) {
  return std::min(hi, std::max(lo, x));
}

py::tuple collapse_interval(
    InArr lo_in, InArr hi_in, py::object prev_obj, py::object amax_obj, double dt) {
  const int n = static_cast<int>(lo_in.size());
  if (n != static_cast<int>(hi_in.size()) || n <= 0) {
    throw std::runtime_error("collapse_interval size mismatch");
  }
  std::vector<double> lo(lo_in.data(), lo_in.data() + n);
  std::vector<double> hi(hi_in.data(), hi_in.data() + n);
  const bool has_prev = !prev_obj.is_none();
  const bool has_amax = !amax_obj.is_none();
  std::vector<double> prev(static_cast<size_t>(n), 0.0);
  std::vector<double> amax(static_cast<size_t>(n), 0.0);
  if (has_prev) {
    const InArr p = prev_obj.cast<InArr>();
    if (static_cast<int>(p.size()) != n) {
      throw std::runtime_error("qdot_prev size mismatch");
    }
    std::memcpy(prev.data(), p.data(), sizeof(double) * static_cast<size_t>(n));
  }
  if (has_amax) {
    const InArr a = amax_obj.cast<InArr>();
    if (static_cast<int>(a.size()) != n) {
      throw std::runtime_error("a_max size mismatch");
    }
    std::memcpy(amax.data(), a.data(), sizeof(double) * static_cast<size_t>(n));
  }
  for (int i = 0; i < n; ++i) {
    if (lo[static_cast<size_t>(i)] <= hi[static_cast<size_t>(i)]) continue;
    const double gap_lo = hi[static_cast<size_t>(i)];
    const double gap_hi = lo[static_cast<size_t>(i)];
    double target = 0.0;
    if (has_prev) {
      target = prev[static_cast<size_t>(i)];
      if (has_amax && dt > 0.0) {
        const double step = amax[static_cast<size_t>(i)] * dt;
        if (target > 0.0) target = std::max(0.0, target - step);
        else if (target < 0.0) target = std::min(0.0, target + step);
      } else {
        target = 0.0;
      }
    }
    double collapsed = clip_d(target, gap_lo, gap_hi);
    if (has_prev && has_amax && dt > 0.0) {
      const double step = amax[static_cast<size_t>(i)] * dt;
      collapsed = clip_d(
          collapsed,
          prev[static_cast<size_t>(i)] - step,
          prev[static_cast<size_t>(i)] + step);
    }
    lo[static_cast<size_t>(i)] = collapsed;
    hi[static_cast<size_t>(i)] = collapsed;
  }
  return py::make_tuple(lo, hi);
}

}  // namespace

PYBIND11_MODULE(_qpik_kernel, m) {
  m.doc() = "C++ QPIK assembly (Eigen internally; no pybind11/eigen.h ABI)";
  m.def("build_wbc_inequalities", &build_wbc_inequalities);
  m.def("singular_values", &singular_values);
  m.def("project_nullspace", &project_nullspace);
  m.def("setup_qp1", &setup_qp1);
  m.def("setup_qp2_costs", &setup_qp2_costs);
  m.def("collapse_interval", &collapse_interval);
}
