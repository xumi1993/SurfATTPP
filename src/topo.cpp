#include "topo.h"
#include "h5io.h"
#include "logger.h"

Topography::Topography(const std::string& filepath)
    : topo_file_(filepath)
{
    read_from_file();
}

void Topography::read_from_file() {
    auto& logger = ATTLogger::logger();
    logger.Info("Reading topography from " + topo_file_, MODULE_TOPO);
    H5IO file(topo_file_, H5IO::RDONLY);
    auto lon_vec = file.read_vector<real_t>("lon");
    auto lat_vec = file.read_vector<real_t>("lat");
    lon_raw = Eigen::Map<Eigen::VectorX<real_t>>(lon_vec.data(), lon_vec.size());
    lat_raw = Eigen::Map<Eigen::VectorX<real_t>>(lat_vec.data(), lat_vec.size());
    topo    = file.read_matrix<real_t>("z");
    topo   /= 1000.0;  // convert from m to km
}

void Topography::bcast() {
    auto& mpi = Parallel::mpi();
    int nx = static_cast<int>(lon_raw.size());
    int ny = static_cast<int>(lat_raw.size());
    mpi.bcast(nx);
    mpi.bcast(ny);
    if (!mpi.is_main()) {
        lon_raw.resize(nx);
        lat_raw.resize(ny);
        topo.resize(nx, ny);
    }
    mpi.bcast(lon_raw.data(), nx);
    mpi.bcast(lat_raw.data(), ny);
    mpi.bcast(topo.data(), nx * ny);
}

Eigen::MatrixX<real_t> Topography::smooth(const real_t sigma) {
    return gaussian_smooth_geo_2(topo, lon_raw, lat_raw, sigma);
}

void Topography::grid(const Eigen::VectorX<real_t>& xgrids,
                      const Eigen::VectorX<real_t>& ygrids) {
    auto& mpi = Parallel::mpi();

    check_bounds(xgrids, ygrids);
    int nx  = static_cast<int>(xgrids.size());
    int ny  = static_cast<int>(ygrids.size());
    dx  = xgrids(1) - xgrids(0);
    dy  = ygrids(1) - ygrids(0);

    if (mpi.is_main()) {
        auto [xx, yy] = meshgrid_ij(xgrids, ygrids);
        z = interp2d(lon_raw, lat_raw, topo, xx, yy);
    } else{
        z.resize(nx, ny);
    }
    mpi.barrier();
    mpi.bcast(z.data(), nx * ny);
}

void Topography::check_bounds(const Eigen::VectorX<real_t>& xgrids,
                               const Eigen::VectorX<real_t>& ygrids) {
    auto &logger = ATTLogger::logger();

    real_t lon_min = lon_raw.minCoeff();
    real_t lon_max = lon_raw.maxCoeff();
    real_t lat_min = lat_raw.minCoeff();
    real_t lat_max = lat_raw.maxCoeff();

    real_t x_min = xgrids.minCoeff();
    real_t x_max = xgrids.maxCoeff();
    real_t y_min = ygrids.minCoeff();
    real_t y_max = ygrids.maxCoeff();

    if (x_min < lon_min || x_max > lon_max ||
        y_min < lat_min || y_max > lat_max) {
        logger.Error(
            std::format("Model grid extends beyond topography bounds: "
                        "lon [{:.2f}, {:.2f}] vs topo [{:.2f}, {:.2f}]; "
                        "lat [{:.2f}, {:.2f}] vs topo [{:.2f}, {:.2f}]",
                        x_min, x_max, lon_min, lon_max,
                        y_min, y_max, lat_min, lat_max),
            MODULE_TOPO
        );
        exit(EXIT_FAILURE);
    }
}

Eigen::MatrixX<real_t> Topography::calc_dip_angle(
    const Eigen::VectorX<real_t>& xgrids,
    const Eigen::VectorX<real_t>& ygrids
) {
    auto &mpi = Parallel::mpi();
    
    if (z.size() == 0) {
        ATTLogger::logger().Error("Topo grid not set. Call grid() first.", MODULE_TOPO);
        exit(EXIT_FAILURE);
    }
    Eigen::MatrixX<real_t> dip_angle;
    if (mpi.is_main()) {
        Eigen::MatrixX<real_t> tx, ty;
        gradient_2_geo(z, xgrids, ygrids, tx, ty);
        dip_angle = (tx.array().square() + ty.array().square()).sqrt().atan();
        dip_angle *= RAD2DEG;
    } else {
        dip_angle(z.rows(), z.cols());
    }
    mpi.barrier();
    mpi.bcast(dip_angle.data(), z.size());
    return dip_angle;
}
