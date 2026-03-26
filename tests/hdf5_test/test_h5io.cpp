#include "h5io.h"
#include "parallel.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void print(const std::string &msg) {
    if (Parallel::mpi().is_main()) std::cout << msg << "\n";
}

template<typename T>
static bool near(T a, T b, T tol = T(1e-9)) { return std::abs(a - b) < tol; }

// ---------------------------------------------------------------------------
// Test: scalar read / write
// ---------------------------------------------------------------------------
static void test_scalar(const std::string &path) {
    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_scalar<int>   ("niter",    42);
        f.write_scalar<double>("step_len", 0.01234);
        f.write_scalar<float> ("eps",      3.14f);
    }
    Parallel::mpi().barrier();

    // All ranks read back
    H5IO f(path, H5IO::RDONLY);
    assert(f.read_scalar<int>   ("niter")    == 42);
    assert(near(f.read_scalar<double>("step_len"), 0.01234));
    assert(near<float>(f.read_scalar<float> ("eps"), 3.14f, 1e-5f));

    print("[PASS] scalar write/read");
}

// ---------------------------------------------------------------------------
// Test: std::vector read / write
// ---------------------------------------------------------------------------
static void test_stdvector(const std::string &path) {
    std::vector<double> depth = {0.0, 5.0, 10.0, 15.0, 20.0};

    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_vector("depth", depth);
    }
    Parallel::mpi().barrier();

    H5IO f(path, H5IO::RDONLY);
    auto d2 = f.read_vector<double>("depth");
    assert(d2.size() == depth.size());
    for (size_t i = 0; i < depth.size(); ++i)
        assert(near(d2[i], depth[i]));

    print("[PASS] std::vector write/read");
}

// ---------------------------------------------------------------------------
// Test: Eigen vector read / write
// ---------------------------------------------------------------------------
static void test_eigen_vector(const std::string &path) {
    Eigen::VectorXd v(5);
    v << 1.1, 2.2, 3.3, 4.4, 5.5;

    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_vector("evec", v);
    }
    Parallel::mpi().barrier();

    H5IO f(path, H5IO::RDONLY);
    auto r = f.read_vector<double>("evec");
    assert(static_cast<int>(r.size()) == v.size());
    for (int i = 0; i < v.size(); ++i)
        assert(near(r[i], v[i]));

    print("[PASS] Eigen vector write/read");
}

// ---------------------------------------------------------------------------
// Test: Eigen matrix read / write (row-major on disk)
// ---------------------------------------------------------------------------
static void test_eigen_matrix(const std::string &path) {
    Eigen::MatrixXd M(3, 4);
    M << 1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12;

    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_matrix("vel", M);
    }
    Parallel::mpi().barrier();

    H5IO f(path, H5IO::RDONLY);
    Eigen::MatrixXd M2 = f.read_matrix<double>("vel");

    assert(M2.rows() == 3 && M2.cols() == 4);
    assert(near((M2 - M).norm(), 0.0, 1e-12));

    print("[PASS] Eigen matrix write/read (3x4)");
}

// ---------------------------------------------------------------------------
// Test: string attribute on root group and on dataset
// ---------------------------------------------------------------------------
static void test_attributes(const std::string &path) {
    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_scalar<int>("dummy", 0);
        f.write_attr("/",       "creator", "SurfATT");
        f.write_attr("/",       "version", "2.0");
        f.write_attr("dummy",   "units",   "none");
    }
    Parallel::mpi().barrier();

    H5IO f(path, H5IO::RDONLY);
    assert(f.read_attr("/",     "creator") == "SurfATT");
    assert(f.read_attr("/",     "version") == "2.0");
    assert(f.read_attr("dummy", "units")   == "none");

    print("[PASS] string attributes (root + dataset)");
}

// ---------------------------------------------------------------------------
// Test: existence check
// ---------------------------------------------------------------------------
static void test_exists(const std::string &path) {
    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_scalar<int>("present", 1);
    }
    Parallel::mpi().barrier();

    H5IO f(path, H5IO::RDONLY);
    assert( f.exists("present"));
    assert(!f.exists("absent"));

    print("[PASS] existence check");
}

// ---------------------------------------------------------------------------
// Test: parallel write — each rank writes its slice; rank 0 reads & sums
// ---------------------------------------------------------------------------
static void test_parallel_write(const std::string &base_path) {
    auto &mpi = Parallel::mpi();
    const int N = 100;

    // Each rank writes its own file with its local slice
    int istart, iend;
    mpi.scatter_range(N, istart, iend);   // 1-based inclusive
    int local_n = iend - istart + 1;

    Eigen::VectorXd local(local_n);
    for (int i = 0; i < local_n; ++i) local(i) = istart + i;   // values 1..N

    std::string rank_path = base_path + ".rank" + std::to_string(mpi.rank()) + ".h5";
    {
        H5IO f(rank_path, H5IO::TRUNC);
        f.write_scalar<int>("istart", istart);
        f.write_scalar<int>("iend",   iend);
        f.write_vector     ("data",   local);
    }
    mpi.barrier();

    // Rank 0 reads all rank files, assembles, checks sum = N*(N+1)/2
    if (mpi.is_main()) {
        double total = 0.0;
        for (int r = 0; r < mpi.size(); ++r) {
            std::string rp = base_path + ".rank" + std::to_string(r) + ".h5";
            H5IO rf(rp, H5IO::RDONLY);
            auto d = rf.read_vector<double>("data");
            for (auto v : d) total += v;
            fs::remove(rp);
        }
        double expected = N * (N + 1.0) / 2.0;
        assert(near(total, expected));
        std::cout << "[PASS] parallel write/read (sum=" << total << ")\n";
    }
    mpi.barrier();
}

// ---------------------------------------------------------------------------
// Test: 3-D volume read / write
// ---------------------------------------------------------------------------
static void test_volume(const std::string &path) {
    const hsize_t NZ = 4, NY = 3, NX = 5;
    const hsize_t total = NZ * NY * NX;

    // Fill with v[iz][iy][ix] = iz*100 + iy*10 + ix
    std::vector<double> ref(total);
    for (hsize_t iz = 0; iz < NZ; ++iz)
        for (hsize_t iy = 0; iy < NY; ++iy)
            for (hsize_t ix = 0; ix < NX; ++ix)
                ref[H5IO::idx3(iz, iy, ix, NY, NX)] =
                    static_cast<double>(iz * 100 + iy * 10 + ix);

    if (Parallel::mpi().is_main()) {
        H5IO f(path, H5IO::TRUNC);
        f.write_volume<double>("vel3d", ref, NZ, NY, NX);
    }
    Parallel::mpi().barrier();

    H5IO f(path, H5IO::RDONLY);
    hsize_t rz = 0, ry = 0, rx = 0;
    auto d = f.read_volume<double>("vel3d", rz, ry, rx);

    assert(rz == NZ && ry == NY && rx == NX);
    assert(d.size() == total);
    for (hsize_t i = 0; i < total; ++i)
        assert(near(d[i], ref[i]));

    // Spot-check a specific voxel: [2][1][3] = 213
    assert(near(d[H5IO::idx3(2, 1, 3, NY, NX)], 213.0));

    print("[PASS] 3-D volume write/read (" +
          std::to_string(NZ) + "x" + std::to_string(NY) + "x" +
          std::to_string(NX) + ")");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    Parallel::init();
    print("=== H5IO tests ===");

    const std::string tmp = "test_h5io_tmp";

    test_scalar      (tmp + "_scalar.h5");
    test_stdvector   (tmp + "_vec.h5");
    test_eigen_vector(tmp + "_evec.h5");
    test_eigen_matrix(tmp + "_mat.h5");
    test_attributes  (tmp + "_attr.h5");
    test_exists      (tmp + "_exists.h5");
    test_volume      (tmp + "_vol.h5");
    test_parallel_write(tmp + "_par");

    // Clean up
    if (Parallel::mpi().is_main()) {
        for (const auto &s : {"_scalar","_vec","_evec","_mat","_attr","_exists","_vol"})
            fs::remove(std::string(tmp) + s + ".h5");
    }

    print("=== All H5IO tests passed ===");
    Parallel::mpi().finalize();
    return 0;
}

