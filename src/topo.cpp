#include "topo.h"
#include "h5io.h"
#include "logger.h"
#include "sph2loc.h"
#include "surfdisp.h"

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
    topo.transposeInPlace();  // HDF5 is row-major, but we want column-major
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

void Topography::smooth(const real_t sigma) {
    z = gaussian_smooth_geo_2(z, lon_raw, lat_raw, sigma);
}

void Topography::grid(const Eigen::VectorX<real_t>& x,
                      const Eigen::VectorX<real_t>& y) {
    auto& mpi = Parallel::mpi();

    check_bounds(x, y);
    int nx  = static_cast<int>(x.size());
    int ny  = static_cast<int>(y.size());
    dx  = x(1) - x(0);
    dy  = y(1) - y(0);
    lon = x;
    lat = y;

    if (mpi.is_main()) {
        auto [xx, yy] = meshgrid_ij(x, y);
        z = interp2d(lon_raw, lat_raw, topo, xx, yy);
    } else{
        z.resize(nx, ny);
    }
    mpi.barrier();
    mpi.bcast(z.data(), nx * ny);
}

void Topography::check_bounds(const Eigen::VectorX<real_t>& x,
                               const Eigen::VectorX<real_t>& y) {
    auto &logger = ATTLogger::logger();

    real_t lon_min = lon_raw.minCoeff();
    real_t lon_max = lon_raw.maxCoeff();
    real_t lat_min = lat_raw.minCoeff();
    real_t lat_max = lat_raw.maxCoeff();

    real_t x_min = x.minCoeff();
    real_t x_max = x.maxCoeff();
    real_t y_min = y.minCoeff();
    real_t y_max = y.maxCoeff();

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

Eigen::MatrixX<real_t> Topography::calc_dip_angle() {
    auto &mpi = Parallel::mpi();
    
    if (z.size() == 0) {
        ATTLogger::logger().Error("Topo grid not set. Call grid() first.", MODULE_TOPO);
        exit(EXIT_FAILURE);
    }
    Eigen::MatrixX<real_t> dip_angle;
    if (mpi.is_main()) {
        Eigen::MatrixX<real_t> tx, ty;
        gradient_2_geo(z, lon, lat, tx, ty);
        dip_angle = (tx.array().square() + ty.array().square()).sqrt().atan();
        dip_angle *= RAD2DEG;
    } else {
        dip_angle(z.rows(), z.cols());
    }
    mpi.barrier();
    mpi.bcast(dip_angle.data(), z.size());
    return dip_angle;
}

void Topography::rotate(
    const real_t xmin, const real_t xmax,
    const real_t ymin, const real_t ymax,
    const real_t clat, const real_t clon, const real_t angle)
{
    auto &mpi = Parallel::mpi();
    auto &logger = ATTLogger::logger();

    dx = lon(1) - lon(0);
    dy = lat(1) - lat(0);
    int nx = static_cast<int>((xmax - xmin) / dx) + 1;
    int ny = static_cast<int>((ymax - ymin) / dy) + 1;
    Eigen::VectorX<real_t> x = Eigen::VectorX<real_t>::LinSpaced(nx, xmin, xmax);
    Eigen::VectorX<real_t> y = Eigen::VectorX<real_t>::LinSpaced(ny, ymin, ymax);
    if (mpi.is_main()) {
        Eigen::MatrixX<real_t>xx_bk, yy_bk;
        auto [xx, yy] = meshgrid_ij(x, y);
        sph2loc::rtp_rotation_reverse(yy, xx, clat, clon, angle, yy_bk, xx_bk);
        if (xx_bk.minCoeff() < lon.minCoeff() || xx_bk.maxCoeff() > lon.maxCoeff() ||
            yy_bk.minCoeff() < lat.minCoeff() || yy_bk.maxCoeff() > lat.maxCoeff()) {
            logger.Error("Rotated grid extends beyond topography bounds.", MODULE_TOPO);
            exit(EXIT_FAILURE);
        }
        z = interp2d(lon, lat, z, xx_bk, yy_bk);
    }
    mpi.barrier();
    if (!mpi.is_main()) z.resize(nx, ny);
    lon = x;
    lat = y;
    mpi.bcast(z.data(), nx * ny);
}

void Topography::copy() {
    auto &mpi = Parallel::mpi();
    if (mpi.is_main()) {
        lon = lon_raw;
        lat = lat_raw;
        z = topo;
    }
    mpi.barrier();
    mpi.bcast(lon.data(), lon.size());
    mpi.bcast(lat.data(), lat.size());
    mpi.bcast(z.data(), z.size());
    dx = lon(1) - lon(0);
    dy = lat(1) - lat(0);
}

void Topography::write(const std::string& filepath) const {
    auto& mpi = Parallel::mpi();
    if (mpi.is_main()) {
        H5IO file(filepath, H5IO::TRUNC);
        file.write_vector("lon", lon);
        file.write_vector("lat", lat);
        file.write_matrix("z", z);
    }
    mpi.barrier();
}