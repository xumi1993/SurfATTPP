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

    // initialize logger (console only)
    ATTLogger::init("surfatt_rotate_topo.log", /*log_level=*/0, /*console_only=*/true);

    auto args = argparse_rotate_topo(argc, argv);
    auto& logger = ATTLogger::logger();
    logger.Info("Rotating topography from " + args.fname + " and saving to " + args.outfname, MODULE_TOPO);

    Topography::read(args.fname);
    auto &topo = Topography::Topo();

    // copy raw topo
    topo.copy();

    topo.rotate(args.xrange[0], args.xrange[1], args.yrange[0], args.yrange[1],
                args.center[0], args.center[1], args.angle);

    topo.z *= 1000.0;  // convert to m

    topo.write(args.outfname);

    return 0;
}
