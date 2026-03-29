#include "topo.h"
#include "input_params.h"
#include "parallel.h"
#include "argparser.h"
#include "logger.h"


int main(int argc, char* argv[])
{
    // parse options
    argparse_rotate_topo(argc, argv);

    // initialise MPI
    Parallel::init();
    auto& mpi = Parallel::mpi();
    if (mpi.size() > 1) {
        mpi.finalize();
        throw std::runtime_error(
            "SURFATT_rotate_topo is not designed for parallel execution. Please run with a single process."
        );
    }

    // initialize logger (console only)
    ATTLogger::init("surfatt_rotate_topo.log", /*log_level=*/0, /*console_only=*/true);

    auto args = argparse_rotate_topo(argc, argv);
    auto& logger = ATTLogger::logger();
    logger.Info("Rotating topography from " + args.fname + " and saving to " + args.outfname, MODULE_TOPO);

    // read topo file and broadcast to all ranks
    Topography::read(args.fname);
    auto &topo = Topography::Topo();

    // copy raw topo
    topo.copy();

    // rotate the topography by the specified angle around the center point, 
    // and interpolate onto the new grid defined by xrange/yrange.
    topo.rotate(args.xrange[0], args.xrange[1], args.yrange[0], args.yrange[1],
                args.center[0], args.center[1], args.angle);
    
    // convert back to m for output
    topo.z *= 1000.0;  // convert to m
    
    // write the rotated topography to an HDF5 file
    topo.write(args.outfname);

    return 0;
}
