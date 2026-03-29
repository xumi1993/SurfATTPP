#include "src_rec.h"
#include "parallel.h"
#include "argparser.h"
#include "sph2loc.h"
#include "logger.h"

int main(int argc, char* argv[])
{   
    auto args = argparse_rotate_src_rec(argc, argv);

    // initialise MPI
    Parallel::init();
    auto& mpi = Parallel::mpi();
    if (mpi.size() > 1) {
        mpi.finalize();
        throw std::runtime_error(
            "SURFATT_rotate_src_rec is not designed for parallel execution. Please run with a single process."
        );
    }

    // logger 
    ATTLogger::init("", /*log_level=*/1, /*console_only=*/true);

    // load source-receiver tables into shared memory
    SrcRec sr;
    sr.load(args.fname);

    // assign stla stlo to Eigen vectors for rotation
    Eigen::VectorX<real_t> stla_vec = Eigen::Map<Eigen::VectorX<real_t>>(sr.stla, sr.n_obs);
    Eigen::VectorX<real_t> stlo_vec = Eigen::Map<Eigen::VectorX<real_t>>(sr.stlo, sr.n_obs);
    Eigen::VectorX<real_t> evla_vec = Eigen::Map<Eigen::VectorX<real_t>>(sr.evla, sr.n_obs);
    Eigen::VectorX<real_t> evlo_vec = Eigen::Map<Eigen::VectorX<real_t>>(sr.evlo, sr.n_obs);

    // rotate source and receiver locations
    auto [new_stla, new_stlo] = sph2loc::rtp_rotation(stla_vec, stlo_vec, args.center[0], args.center[1], args.angle);
    auto [new_evla, new_evlo] = sph2loc::rtp_rotation(evla_vec, evlo_vec, args.center[0], args.center[1], args.angle);

    // copy back to shared memory
    std::copy(new_stla.begin(), new_stla.end(), sr.stla);
    std::copy(new_stlo.begin(), new_stlo.end(), sr.stlo);
    std::copy(new_evla.begin(), new_evla.end(), sr.evla);
    std::copy(new_evlo.begin(), new_evlo.end(), sr.evlo);

    sr.write(args.outfname);

    return 0;
}