#include "config.h"
#include "input_params.h"
#include "parallel.h"
#include "logger.h"
#include "model_grid.h"
#include "src_rec.h"
#include "surf_grid.h"
#include "argparser.h"
#include "decomposer.h"
#include "inversion.h"


int main(int argc, char* argv[]) {

    // parse options
    auto args = argparse_cb_fwd(argc, argv);

    // initialise MPI
    Parallel::init();

    // read input parameters
    InputParams::read(input_file);
    auto& IP = InputParams::IP();

    // initialize logger
    ATTLogger::init(
        fmt::format("{}/{}", IP.output().output_path, LOG_FNAME),
        IP.output().log_level,
        false
    );

    // load source-receiver tables into shared memory
    for (auto [wt, vt] : IP.data().active_data)
        SrcRec::SR(wt, vt).load(IP.data().file_of(wt, vt));
    SrcRec::build_stas();

    // build model grid
    ModelGrid::init();
    auto &mg = ModelGrid::MG();

    // build initial model
    mg.build_init_model();

    // add checkerboard perturbation if requested
    mg.add_perturbation(
        args.ncb[0], args.ncb[1], args.ncb[2],
        args.pert_vel, args.hmarg, args.anom_size, args.only_vs
    );
    mg.add_aniso_perturbation(
        args.ncb_ani[0], args.ncb_ani[1], args.ncb_ani[2],
        args.ani_angle, args.pert_ani, args.hmarg, args.anom_size
    );
    mg.write(TARGET_MODEL_FNAME);

    // initialize decomposer (for parallel execution)
    Decomposer::DCP();

    // load topography if needed
    if (IP.topo().is_consider_topo) {
        // load topography
        Topography::read(IP.topo().topo_file);
    }
    // build surface grid and compute reference travel times
    for (auto [wt, vt] : IP.data().active_data)
        SurfGrid::SG(wt, vt).build_media();

    // run forward simulation
    Inversion::init();
    Inversion::INV().run_forward();

    // free shared memory windows
    for (auto [wt, vt] : IP.data().active_data)
        SrcRec::SR(wt, vt).release_shm();

    return 0;

}