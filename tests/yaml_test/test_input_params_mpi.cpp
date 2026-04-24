#include "input_params.h"
#include "parallel.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>

static void assert_near(double a, double b, double tol = 1e-9) {
    assert(std::abs(a - b) < tol);
}

// Print helper: only rank 0 writes to stdout
static void print(const std::string &msg) {
    if (Parallel::mpi().is_main()) std::cout << msg << "\n";
}

// ---------------------------------------------------------------------------
// Verification — every rank runs these checks on its own copy of the params
// ---------------------------------------------------------------------------
static void verify_all() {
    const auto &d   = InputParams::IP().data();
    const auto &dom = InputParams::IP().domain();
    const auto &t   = InputParams::IP().topo();
    const auto &inv = InputParams::IP().inversion();

    // data
    assert(d.src_rec_file_ph == "src_rec_file_rotated.csv");
    assert(d.src_rec_file_gr.empty());
    assert(d.iwave == 2);
    assert(d.vel_type[0] == true  && d.vel_type[1] == false);
    assert_near(d.weights[0], 0.5);
    assert_near(d.weights[1], 0.5);

    // output
    assert(o.output_path   == "OUTPUT_FILES/");
    assert(o.log_level     == 1);

    // domain
    assert_near(dom.depth[0], 0.0);
    assert_near(dom.depth[1], 15.0);
    assert_near(dom.interval[0], 0.01);
    assert_near(dom.interval[2], 0.5);
    assert(dom.num_grid_margin == 5);

    // topo
    assert(t.is_consider_topo == true);
    assert(t.topo_file        == "hawaii_rotated.nc");
    assert_near(t.wavelen_factor, 2.5);

    // inversion
    assert(inv.use_alpha_beta_rho == true);
    assert(inv.rho_scaling        == true);
    assert(inv.niter        == 40);
    assert_near(inv.min_derr,    0.0001);
    assert(inv.optim_method  == 2);
    assert_near(inv.step_length, 0.02);
    assert_near(inv.maxshrink,   0.6);
    assert(inv.max_sub_niter == 10);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // 1. Boot MPI
    Parallel::init();
    auto &mpi = Parallel::mpi();

    const std::string yaml_file =
        (argc > 1) ? argv[1] : "tests/yaml_test/input_params.yml";

    // 2. Read on rank 0, broadcast to all other ranks automatically
    InputParams::read(yaml_file);

    // Show which ranks participated
    std::ostringstream oss;
    oss << "[rank " << mpi.rank() << "/" << mpi.size()
        << "] received params — niter=" << InputParams::IP().inversion().niter
        << "  output_path=" << InputParams::IP().output().output_path;
    // Each rank prints; flush order depends on MPI implementation
    for (int r = 0; r < mpi.size(); ++r) {
        mpi.barrier();
        if (mpi.rank() == r) std::cout << oss.str() << "\n" << std::flush;
    }
    mpi.barrier();

    // 3. Every rank verifies its own copy
    verify_all();

    // Collect pass/fail: each rank contributes 1 if OK (assert would have
    // aborted otherwise), sum must equal mpi.size()
    int local_ok = 1, total_ok = 0;
    mpi.sum_all_all(local_ok, total_ok);

    print("\n[PASS] all " + std::to_string(total_ok) + "/" +
          std::to_string(mpi.size()) + " ranks verified params correctly.");

    mpi.finalize();
    return 0;
}
