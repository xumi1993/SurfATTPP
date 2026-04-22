#include "input_params.h"
#include "parallel.h"
#include "src_rec.h"
#include "logger.h"
#include "config.h"
#include "utils.h"
#include "model_grid.h"
#include "argparser.h"
#include "topo.h"
#include "surf_grid.h"
#include "preproc.h"
#include "decomposer.h"
#include "inversion.h"

#include <algorithm>


int main(int argc, char* argv[])
{
    // parse options    
    argparse_tomo(argc, argv);

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
    if (IP.data().vel_type[0]) SrcRec::SR_ph().load(IP.data().src_rec_file_ph);
    if (IP.data().vel_type[1]) SrcRec::SR_gr().load(IP.data().src_rec_file_gr);
    SrcRec::build_stas();

    // build model grid
    ModelGrid::init();
    auto &mg = ModelGrid::MG();

    // build initial model
    mg.build_init_model();

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

    Inversion::init();

    // run forward and adjoint calculations
    if (run_mode == FORWARD_ONLY) {
        Inversion::INV().run_forward();
    } else {
        Inversion::INV().run_inversion();
    }

    SrcRec::SR_ph().release_shm();
    SrcRec::SR_gr().release_shm();

    double total_time = Parallel::mpi().elapsed_time();
    ATTLogger::logger().Info(fmt::format("Total runtime: {:.2f} seconds", total_time), MODULE_MAIN);

    return 0;
}