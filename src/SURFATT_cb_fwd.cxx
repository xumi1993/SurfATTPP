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
        std::format("{}/{}", IP.output().output_path, LOG_FNAME),
        IP.output().log_level,
        false
    );

    // load source-receiver tables into shared memory
    if (IP.data().vel_type[0]) SrcRec::SR_ph().load(IP.data().src_rec_file_ph);
    if (IP.data().vel_type[1]) SrcRec::SR_gr().load(IP.data().src_rec_file_gr);
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
        args.ani_angle, args.pert_ani, args.hmarg, _0_CR
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
    if (IP.data().vel_type[0]) SurfGrid::SG_ph().build_media();
    if (IP.data().vel_type[1]) SurfGrid::SG_gr().build_media();

    // run forward simulation
    Inversion::init();
    Inversion::INV().run_forward();

    // free shared memory windows
    SrcRec::SR_ph().release_shm();
    SrcRec::SR_gr().release_shm();

    return 0;

}