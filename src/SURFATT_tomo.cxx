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

    // load topography if needed
    if (IP.topo().is_consider_topo) {
        // load topography
        Topography::read(IP.topo().topo_file);
    }
    // build surface grid and compute reference travel times
    if (IP.data().vel_type[0]) SurfGrid::SG_ph().build_media();
    if (IP.data().vel_type[1]) SurfGrid::SG_gr().build_media();

    // run forward and adjoint calculations
    if (FORWARD_ONLY) {
        preproc::run_forward_adjoint(false);
    } else {
        preproc::run_forward_adjoint(true);
    }

    SrcRec::SR_ph().release_shm();
    SrcRec::SR_gr().release_shm();

    return 0;
}