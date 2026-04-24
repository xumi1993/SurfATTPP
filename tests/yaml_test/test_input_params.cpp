#include "input_params.h"
#include "parallel.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

static void assert_near(double a, double b, double tol = 1e-9) {
    assert(std::abs(a - b) < tol);
}

// ---------------------------------------------------------------------------
// Simulate how other modules would use the singleton.
// In a real project these would live in separate .cpp files and only
// #include "input_params.h" — no parameter passing needed.
// ---------------------------------------------------------------------------

static void check_data() {
    // Any file can call InputParams::IP() after init() has been called.
    const auto &d = InputParams::IP().data();
    assert(d.src_rec_file_rl_ph == "src_rec_file_rotated.csv");
    assert(d.src_rec_file_rl_gr.empty());
    assert(d.src_rec_file_lv_ph.empty());
    assert(d.src_rec_file_lv_gr.empty());
    assert(d.wave_type[0] == true  && d.wave_type[1] == false);
    assert(d.vel_type[0]  == true  && d.vel_type[1]  == false);
    assert(d.active_data.size() == 1);
    assert(d.active_data[0].first  == WaveType::RL);
    assert(d.active_data[0].second == surfType::PH);
    assert_near(d.weights[0], 0.5);
    assert_near(d.weights[1], 0.5);
    std::cout << "[PASS] data\n";
}

static void check_output() {
    const auto &o = InputParams::IP().output();
    assert(o.output_path   == "OUTPUT_FILES/");
    assert(o.log_level     == 1);
    std::cout << "[PASS] output\n";
}

static void check_domain() {
    const auto &dom = InputParams::IP().domain();
    assert_near(dom.depth[0], 0.0);
    assert_near(dom.depth[1], 15.0);
    assert_near(dom.interval[0], 0.01);
    assert_near(dom.interval[1], 0.01);
    assert_near(dom.interval[2], 0.5);
    assert(dom.num_grid_margin == 5);
    std::cout << "[PASS] domain\n";
}

static void check_topo() {
    const auto &t = InputParams::IP().topo();
    assert(t.is_consider_topo == true);
    assert(t.topo_file        == "hawaii_rotated.nc");
    assert_near(t.wavelen_factor, 2.5);
    std::cout << "[PASS] topo\n";
}

static void check_inversion() {
    const auto &inv = InputParams::IP().inversion();
    assert(inv.use_alpha_beta_rho == true);
    assert(inv.rho_scaling        == true);
    assert(inv.niter        == 40);
    assert_near(inv.min_derr, 0.0001);
    assert(inv.optim_method  == 2);
    assert_near(inv.step_length, 0.02);
    assert_near(inv.maxshrink,   0.6);
    assert(inv.max_sub_niter == 10);
    std::cout << "[PASS] inversion\n";
}

static void check_generic_accessors() {
    auto &p = InputParams::IP();
    // get<T> with dot-notation key
    assert(p.get<int>("inversion.niter") == 40);
    // get<T> with default: key exists → returns actual value
    assert(p.get<int>("inversion.ncomponents", 99) == 5);
    // get<T> with default: key absent → returns default
    assert(p.get<int>("inversion.nonexistent", 99) == 99);
    // has()
    assert(p.has("topo.topo_file")   == true);
    assert(p.has("topo.nonexistent") == false);
    std::cout << "[PASS] generic get/has\n";
}

static void check_error_before_init() {
    // Calling IP() before init() must throw.
    // (Re-create a fresh pointer state via a local scope isn't possible with
    //  the singleton, so we just verify the missing-key exception path.)
    bool caught = false;
    try {
        InputParams::IP().get<int>("no.such.key");
    } catch (const std::runtime_error &) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] missing-key exception\n";
}

// ---------------------------------------------------------------------------
// main: init once, then call each module's check function
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    const std::string yaml_file =
        (argc > 1) ? argv[1] : "tests/input_params.yml";

    // 1. Boot MPI (needed for bcast in InputParams)
    Parallel::init();

    // ---- One-time initialisation (done in main, e.g. at program startup) --
    InputParams::read(yaml_file);

    // ---- Each "module" accesses the singleton independently ----------------
    check_data();
    check_output();
    check_domain();
    check_topo();
    check_inversion();
    check_generic_accessors();
    check_error_before_init();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
