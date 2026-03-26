#pragma once

#include "config.h"

#include <H5Cpp.h>
#include <Eigen/Core>

#include <string>
#include <vector>
#include <stdexcept>
#include <type_traits>

// ---------------------------------------------------------------------------
// Compile-time HDF5 type trait
// ---------------------------------------------------------------------------
template<typename T> inline H5::PredType h5_type_of();
template<> inline H5::PredType h5_type_of<int>()           { return H5::PredType::NATIVE_INT; }
template<> inline H5::PredType h5_type_of<float>()         { return H5::PredType::NATIVE_FLOAT; }
template<> inline H5::PredType h5_type_of<real_t>()        { return H5::PredType::NATIVE_DOUBLE; }
template<> inline H5::PredType h5_type_of<real2_t>()       { return H5::PredType::NATIVE_LDOUBLE; }

// ---------------------------------------------------------------------------
// H5IO — thin RAII wrapper around H5::H5File
//
// Usage (write):
//   H5IO f("model.h5", H5IO::TRUNC);
//   f.write_scalar("niter", 40);
//   f.write_vector("depth", v);
//   f.write_matrix("vel",   M);
//   f.write_volume("rho",   data, nz, ny, nx);
//   f.write_attr  ("/",     "creator", "SurfATT");
//
// Usage (read):
//   H5IO f("model.h5", H5IO::RDONLY);
//   int niter = f.read_scalar<int>("niter");
//   auto v    = f.read_vector<double>("depth");
//   auto M    = f.read_matrix<double>("vel");     // returns MatrixXd (col-major)
//   hsize_t nz,ny,nx;
//   auto d    = f.read_volume<double>("rho", nz, ny, nx);  // flat [nz*ny*nx]
// ---------------------------------------------------------------------------
class H5IO {
public:
    enum Mode { RDONLY, RDWR, TRUNC };

    explicit H5IO(const std::string &path, Mode mode = RDONLY);
    ~H5IO() = default;

    // ---- Scalar ------------------------------------------------------------
    template<typename T>
    void write_scalar(const std::string &name, T value) {
        ensure_not_readonly();
        hsize_t dims[1] = {1};
        H5::DataSpace sp(1, dims);
        H5::DataSet ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(&value, h5_type_of<T>());
    }

    template<typename T>
    T read_scalar(const std::string &name) const {
        H5::DataSet ds = file_.openDataSet(name);
        T value{};
        ds.read(&value, h5_type_of<T>());
        return value;
    }

    // ---- 1-D vector (std::vector<T> or Eigen vector) ----------------------
    template<typename T>
    void write_vector(const std::string &name, const std::vector<T> &v) {
        ensure_not_readonly();
        hsize_t dims[1] = {static_cast<hsize_t>(v.size())};
        H5::DataSpace sp(1, dims);
        H5::DataSet ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(v.data(), h5_type_of<T>());
    }

    template<typename Derived>
    void write_vector(const std::string &name,
                      const Eigen::DenseBase<Derived> &v) {
        using T = typename Derived::Scalar;
        ensure_not_readonly();
        // Force evaluation to a contiguous column-vector
        Eigen::Matrix<T, Eigen::Dynamic, 1> tmp = v;
        hsize_t dims[1] = {static_cast<hsize_t>(tmp.size())};
        H5::DataSpace sp(1, dims);
        H5::DataSet ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(tmp.data(), h5_type_of<T>());
    }

    template<typename T>
    std::vector<T> read_vector(const std::string &name) const {
        H5::DataSet   ds = file_.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        hsize_t n = 0;
        sp.getSimpleExtentDims(&n, nullptr);
        std::vector<T> v(n);
        ds.read(v.data(), h5_type_of<T>());
        return v;
    }

    // ---- 2-D matrix (Eigen, stored row-major on disk) ----------------------
    template<typename T>
    void write_matrix(const std::string &name,
                      const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &M) {
        ensure_not_readonly();
        // Write as row-major so dimensions match intuition (rows × cols)
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Rm = M;
        hsize_t dims[2] = {static_cast<hsize_t>(M.rows()),
                            static_cast<hsize_t>(M.cols())};
        H5::DataSpace sp(2, dims);
        H5::DataSet   ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(Rm.data(), h5_type_of<T>());
    }

    template<typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    read_matrix(const std::string &name) const {
        H5::DataSet   ds = file_.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        hsize_t dims[2]  = {0, 0};
        sp.getSimpleExtentDims(dims, nullptr);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Rm(dims[0], dims[1]);
        ds.read(Rm.data(), h5_type_of<T>());
        return Rm;  // implicitly converts to ColMajor if assigned to MatrixX*
    }

    // ---- 3-D volume (flat array, dims: nz × ny × nx, row-major on disk) ---
    // data must have exactly nz*ny*nx elements.
    template<typename T>
    void write_volume(const std::string &name,
                      const std::vector<T> &data,
                      hsize_t nz, hsize_t ny, hsize_t nx) {
        ensure_not_readonly();
        if (data.size() != nz * ny * nx)
            throw std::invalid_argument(
                "H5IO::write_volume: data.size() != nz*ny*nx");
        hsize_t dims[3] = {nz, ny, nx};
        H5::DataSpace sp(3, dims);
        H5::DataSet   ds = file_.createDataSet(name, h5_type_of<T>(), sp);
        ds.write(data.data(), h5_type_of<T>());
    }

    // Read a 3-D dataset; nz/ny/nx are set to the actual dataset dimensions.
    template<typename T>
    std::vector<T> read_volume(const std::string &name,
                               hsize_t &nz, hsize_t &ny, hsize_t &nx) const {
        H5::DataSet   ds = file_.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        if (sp.getSimpleExtentNdims() != 3)
            throw std::runtime_error(
                "H5IO::read_volume: dataset '" + name + "' is not 3-D");
        hsize_t dims[3] = {0, 0, 0};
        sp.getSimpleExtentDims(dims, nullptr);
        nz = dims[0]; ny = dims[1]; nx = dims[2];
        std::vector<T> data(nz * ny * nx);
        ds.read(data.data(), h5_type_of<T>());
        return data;
    }

    // Convenience: index helper for flat 3-D arrays (row-major)
    // Usage: data[idx3(iz, iy, ix, ny, nx)]
    static constexpr hsize_t idx3(hsize_t iz, hsize_t iy, hsize_t ix,
                                   hsize_t ny, hsize_t nx) noexcept {
        return iz * ny * nx + iy * nx + ix;
    }

    // ---- String attribute on a group/dataset -------------------------------
    void write_attr(const std::string &obj_path,
                    const std::string &attr_name,
                    const std::string &value);

    std::string read_attr(const std::string &obj_path,
                          const std::string &attr_name) const;

    // ---- Existence check ---------------------------------------------------
    bool exists(const std::string &name) const;

    // ---- Underlying file handle (for advanced use) -------------------------
    H5::H5File &file() { return file_; }
    const H5::H5File &file() const { return file_; }

private:
    H5::H5File file_;
    bool       readonly_{false};

    void ensure_not_readonly() const {
        if (readonly_)
            throw std::logic_error("H5IO: file opened read-only");
    }
};
