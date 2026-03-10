// pybind11 bindings for the C++ LOB engine.
// Exposes LOBEngine to Python and a fast batch_reconstruct that returns
// a pandas-ready dict of numpy arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lob_engine.hpp"

namespace py = pybind11;

// Wrapper that accepts numpy arrays and returns a dict of numpy arrays
// ready for pd.DataFrame construction.
static py::dict py_batch_reconstruct(py::array_t<int64_t> timestamps,
                                     py::array_t<int32_t> types,
                                     py::array_t<int32_t> sides,
                                     py::array_t<double> prices,
                                     py::array_t<double> qtys,
                                     py::array_t<int64_t> update_ids,
                                     int n_levels) {
    auto ts_buf = timestamps.request();
    auto ty_buf = types.request();
    auto si_buf = sides.request();
    auto pr_buf = prices.request();
    auto qt_buf = qtys.request();
    auto ui_buf = update_ids.request();

    int64_t n = ts_buf.shape[0];

    auto [col_names, data] = LOBEngine::batch_reconstruct(
        static_cast<const int64_t*>(ts_buf.ptr),
        static_cast<const int*>(ty_buf.ptr),
        static_cast<const int*>(si_buf.ptr),
        static_cast<const double*>(pr_buf.ptr),
        static_cast<const double*>(qt_buf.ptr),
        static_cast<const int64_t*>(ui_buf.ptr), n, n_levels);

    int n_cols = static_cast<int>(col_names.size());
    int64_t n_rows = (n_cols > 0) ? static_cast<int64_t>(data.size()) / n_cols : 0;

    py::dict result;
    for (int c = 0; c < n_cols; ++c) {
        // Each column is a contiguous slice of the flat data array
        py::array_t<double> arr(n_rows);
        auto arr_buf = arr.mutable_unchecked<1>();
        for (int64_t r = 0; r < n_rows; ++r) {
            arr_buf(r) = data[static_cast<size_t>(r) +
                              static_cast<size_t>(c) * n_rows];
        }

        // timestamp column should be int64
        if (col_names[c] == "timestamp") {
            py::array_t<int64_t> int_arr(n_rows);
            auto int_buf = int_arr.mutable_unchecked<1>();
            for (int64_t r = 0; r < n_rows; ++r) {
                int_buf(r) = static_cast<int64_t>(arr_buf(r));
            }
            result[py::cast(col_names[c])] = int_arr;
        } else {
            result[py::cast(col_names[c])] = arr;
        }
    }

    return result;
}

PYBIND11_MODULE(_lob_cpp, m) {
    m.doc() = "C++ LOB reconstruction engine for high-performance order book "
              "processing (1M+ updates/sec target)";

    py::class_<LOBEngine>(m, "LOBEngine")
        .def(py::init<>())
        .def("update", &LOBEngine::update, py::arg("side"), py::arg("price"),
             py::arg("qty"),
             "Apply a single price-level update. qty <= 0 removes the level.")
        .def("apply_snapshot", &LOBEngine::apply_snapshot, py::arg("side"),
             py::arg("levels"),
             "Replace one side of the book with a full snapshot.")
        .def("best_bid", &LOBEngine::best_bid,
             "Best bid as (price, qty). Returns (NaN, NaN) if empty.")
        .def("best_ask", &LOBEngine::best_ask,
             "Best ask as (price, qty). Returns (NaN, NaN) if empty.")
        .def("mid_price", &LOBEngine::mid_price,
             "Mid-price = (best_bid + best_ask) / 2. NaN if empty.")
        .def("spread", &LOBEngine::spread,
             "Spread = best_ask - best_bid. NaN if empty.")
        .def("top_n", &LOBEngine::top_n, py::arg("side"), py::arg("n") = 10,
             "Top N levels for a side, padded with (NaN, NaN).")
        .def_property("last_update_ts", &LOBEngine::last_update_ts,
                       &LOBEngine::set_last_update_ts);

    m.def("batch_reconstruct", &py_batch_reconstruct, py::arg("timestamps"),
          py::arg("types"), py::arg("sides"), py::arg("prices"),
          py::arg("qtys"), py::arg("update_ids"), py::arg("n_levels") = 10,
          "Process event arrays in C++ and return a dict of numpy arrays "
          "ready for pd.DataFrame().");
}
