#include "h5io.h"
#include "argparser.h"
#include "rapidcsv.h"
#include "sph2loc.h"
#include "logger.h"
#include "parallel.h"
#include "utils.h"

#include <cmath>
#include <format>


int main(int argc, char* argv[])
{   
    auto args = argparse_rotate_model(argc, argv);

    // initialise MPI
    Parallel::init();
    auto& mpi = Parallel::mpi();

    // logger
    ATTLogger::init("", /*log_level=*/2, /*console_only=*/true);
    auto &logger = ATTLogger::logger();

    if (mpi.size() > 1) {
        logger.Error("SURFATT_rotate_model is not designed for parallel execution. Please run with a single process.", MODULE_MAIN);
        mpi.finalize();
        return EXIT_FAILURE;
    }
    logger.Info("Rotating model and exporting CSV...", MODULE_MAIN);

    // read model from hdf5 file
    H5IO file(args.fname, H5IO::RDONLY);
    auto x_vec = file.read_vector<real_t>("x");
    auto y_vec = file.read_vector<real_t>("y");
    auto z_vec = file.read_vector<real_t>("z");
    int ngrid_i = static_cast<int>(x_vec.size());
    int ngrid_j = static_cast<int>(y_vec.size());
    int ngrid_k = static_cast<int>(z_vec.size());
    int nelem = ngrid_i * ngrid_j * ngrid_k;
    logger.Info(std::format("Loaded model grid: {} x {} x {}", ngrid_i, ngrid_j, ngrid_k), MODULE_MAIN);

    // read model properties
    hsize_t ni = 0, nj = 0, nk = 0;
    auto vs = file.read_volume<real_t>("vs", ni, nj, nk);

    std::vector<real_t> g0;
    std::vector<real_t> theta;

    const bool is_aniso = file.exists("g0") && file.exists("theta");
    if (is_aniso) {
        hsize_t ai = 0, aj = 0, ak = 0;
        g0 = file.read_volume<real_t>("g0", ai, aj, ak);
        theta = file.read_volume<real_t>("theta", ai, aj, ak);
    }
    
    // flatten axises for rotation

    std::vector<real_t> x_flat(nelem);
    std::vector<real_t> y_flat(nelem);
    std::vector<real_t> z_flat(nelem);
    for (int i = 0; i < ngrid_i; ++i) {
        for (int j = 0; j < ngrid_j; ++j) {
            for (int k = 0; k < ngrid_k; ++k) {
                x_flat[I2V(i, j, k)] = x_vec[i];
                y_flat[I2V(i, j, k)] = y_vec[j];
                z_flat[I2V(i, j, k)] = z_vec[k];
            }
        }
    }

    // assign x/y grids to Eigen vectors for rotation
    Eigen::VectorX<real_t> x_ev = Eigen::Map<Eigen::VectorX<real_t>>(x_flat.data(), nelem);
    Eigen::VectorX<real_t> y_ev = Eigen::Map<Eigen::VectorX<real_t>>(y_flat.data(), nelem);
    Eigen::VectorX<real_t> new_x_ev = Eigen::VectorX<real_t>(nelem);
    Eigen::VectorX<real_t> new_y_ev = Eigen::VectorX<real_t>(nelem);

    if (!args.center.has_value()) {
        logger.Error("Rotation center is required: please provide -c clat/clon", MODULE_MAIN);
        mpi.finalize();
        return EXIT_FAILURE;
    }

    sph2loc::rtp_rotation_reverse(
        y_ev, x_ev,
        args.center.value()[0], args.center.value()[1], -args.angle,
        new_y_ev, new_x_ev
    );

    if (is_aniso) {
        for (int i = 0; i < nelem; ++i) {
            theta[i] = std::fmod(theta[i] + args.angle, static_cast<real_t>(360.0));
        }
    }
    logger.Info("Rotation completed.", MODULE_MAIN);

    std::vector<real_t> new_x(new_x_ev.data(), new_x_ev.data() + nelem);
    std::vector<real_t> new_y(new_y_ev.data(), new_y_ev.data() + nelem);

    // write to csv file
    rapidcsv::Document doc;
    doc.SetColumn<std::string>(0, fmt_col(new_x.data(), nelem, 4));
    doc.SetColumn<std::string>(1, fmt_col(new_y.data(), nelem, 4));
    doc.SetColumn<std::string>(2, fmt_col(z_flat.data(), nelem, 4));
    doc.SetColumn<std::string>(3, fmt_col(vs.data(), nelem, 6));
    doc.SetColumnName(0, "lon");
    doc.SetColumnName(1, "lat");
    doc.SetColumnName(2, "depth");
    doc.SetColumnName(3, "vs");
    if (is_aniso) {
        doc.SetColumn<std::string>(4, fmt_col(g0.data(), nelem, 6));
        doc.SetColumn<std::string>(5, fmt_col(theta.data(), nelem, 6));
        doc.SetColumnName(4, "g0");
        doc.SetColumnName(5, "theta");
    }
    doc.Save(args.outfname);
    logger.Info(std::format("CSV written to {}", args.outfname), MODULE_MAIN);
    return 0;
}