#include "parallel.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

static void print(const std::string &msg) {
    if (Parallel::mpi().is_main()) std::cout << msg << "\n";
}

template<typename T>
static bool near(T a, T b, T tol = T(1e-9)) { return std::abs(a - b) < tol; }

// ---------------------------------------------------------------------------
// Test 1: each rank builds a local Eigen vector, sum-allreduce → global sum
// ---------------------------------------------------------------------------
static void test_vector_allreduce() {
    auto &mpi = Parallel::mpi();
    const int N = 6;

    // rank r contributes vec = (r+1) * [1,2,3,4,5,6]
    Eigen::VectorXd local(N);
    for (int i = 0; i < N; ++i) local(i) = (mpi.rank() + 1) * (i + 1.0);

    Eigen::VectorXd global = Eigen::VectorXd::Zero(N);
    // sum_all_all works on raw pointers — Eigen guarantees contiguous storage
    mpi.sum_all_all(local.data(), global.data(), N);

    // expected: sum_{r=0}^{size-1}(r+1) * [1..N]
    int sum_ranks = mpi.size() * (mpi.size() + 1) / 2;
    for (int i = 0; i < N; ++i)
        assert(near(global(i), sum_ranks * (i + 1.0)));

    print("[PASS] vector allreduce (sum)");
}

// ---------------------------------------------------------------------------
// Test 2: rank 0 broadcasts an Eigen matrix to all ranks
// ---------------------------------------------------------------------------
static void test_matrix_bcast() {
    auto &mpi = Parallel::mpi();

    Eigen::MatrixXd M(3, 3);
    if (mpi.is_main()) {
        M << 1, 2, 3,
             4, 5, 6,
             7, 8, 9;
    } else {
        M.setZero();
    }

    // MatrixXd is column-major and contiguous
    mpi.bcast(M.data(), static_cast<int>(M.size()));

    assert(near(M(0, 0), 1.0) && near(M(2, 1), 8.0) && near(M(2, 2), 9.0));
    assert(near(M.sum(), 45.0));

    print("[PASS] matrix broadcast");
}

// ---------------------------------------------------------------------------
// Test 3: each rank computes a partial dot product; sum-allreduce assembles
//         the full dot product distributed across ranks
// ---------------------------------------------------------------------------
static void test_distributed_dot() {
    auto &mpi = Parallel::mpi();
    const int total = 100;

    // scatter [0..total) across ranks
    int istart, iend;
    mpi.scatter_range(total, istart, iend);  // 1-based inclusive

    // build local slices of u = [1..100], v = [1..100]
    int local_n = iend - istart + 1;
    Eigen::VectorXd u(local_n), v(local_n);
    for (int i = 0; i < local_n; ++i) {
        double idx = istart + i;
        u(i) = idx;
        v(i) = idx;
    }

    double local_dot  = u.dot(v);
    double global_dot = 0.0;
    mpi.sum_all_all(local_dot, global_dot);

    // sum_{i=1}^{100} i^2 = 100*101*201/6 = 338350
    assert(near(global_dot, 338350.0));

    print("[PASS] distributed dot product");
}

// ---------------------------------------------------------------------------
// Test 4: rank 0 solves a linear system; broadcasts solution; all ranks verify
// ---------------------------------------------------------------------------
static void test_linear_solve_bcast() {
    auto &mpi = Parallel::mpi();

    Eigen::VectorXd x(3);
    if (mpi.is_main()) {
        Eigen::Matrix3d A;
        A << 2, 1, 0,
             1, 3, 1,
             0, 1, 2;
        Eigen::Vector3d b(1, 2, 3);
        x = A.ldlt().solve(b);
    } else {
        x.setZero();
    }

    mpi.bcast(x.data(), 3);

    // Verify A*x == b on every rank
    Eigen::Matrix3d A;
    A << 2, 1, 0,
         1, 3, 1,
         0, 1, 2;
    Eigen::Vector3d b(1, 2, 3);
    Eigen::Vector3d residual = A * x - b;
    assert(near(residual.norm(), 0.0, 1e-10));

    print("[PASS] linear solve + broadcast");
}

// ---------------------------------------------------------------------------
// Test 5: maxloc_all — find the rank holding the global maximum of a vector
// ---------------------------------------------------------------------------
static void test_maxloc() {
    auto &mpi = Parallel::mpi();

    // Each rank holds a scalar: rank r → value = r * 10.0
    double local_val = mpi.rank() * 10.0;

    double best_val; int best_rank;
    mpi.maxloc_all<double>(local_val, mpi.rank(), best_val, best_rank);

    assert(best_rank == mpi.size() - 1);
    assert(near(best_val, (mpi.size() - 1) * 10.0));

    print("[PASS] maxloc_all");
}

// ---------------------------------------------------------------------------
// Test 6: gather an Eigen vector from all ranks onto rank 0
// ---------------------------------------------------------------------------
static void test_vector_gather() {
    auto &mpi = Parallel::mpi();
    const int local_n = 3;

    // Each rank r sends [r*local_n+0, r*local_n+1, r*local_n+2]
    Eigen::VectorXd local(local_n);
    for (int i = 0; i < local_n; ++i)
        local(i) = mpi.rank() * local_n + i;

    Eigen::VectorXd gathered;
    if (mpi.is_main()) gathered.resize(mpi.size() * local_n);

    mpi.gather_all(local.data(), local_n,
                   mpi.is_main() ? gathered.data() : nullptr, local_n);

    if (mpi.is_main()) {
        for (int r = 0; r < mpi.size(); ++r)
            for (int i = 0; i < local_n; ++i)
                assert(near(gathered(r * local_n + i), double(r * local_n + i)));
        std::cout << "[PASS] vector gather\n";
    }
    mpi.barrier();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    Parallel::init();
    auto &mpi = Parallel::mpi();

    print("=== Eigen + MPI tests (ranks: " + std::to_string(mpi.size()) + ") ===");

    test_vector_allreduce();
    test_matrix_bcast();
    test_distributed_dot();
    test_linear_solve_bcast();
    test_maxloc();
    test_vector_gather();

    print("=== All Eigen + MPI tests passed ===");

    mpi.finalize();
    return 0;
}
