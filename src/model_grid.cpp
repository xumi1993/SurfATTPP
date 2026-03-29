#include "model_grid.h"
#include "src_rec.h"
#include "logger.h"
#include "inversion1d.h"
#include "h5io.h"
#include "minpack.hpp"
#include "utils.h"

#include <array>
#include <cmath>
#include <vector>


namespace {
    // Compute the bounding box (lon/lat min/max) of the station network
    // by scanning all station coordinates from the merged station table.
    void get_domain_min_max(real_t &lon_min, real_t &lon_max, real_t &lat_min, real_t &lat_max) {
        Eigen::VectorX<real_t> stla_vec = Eigen::Map<Eigen::VectorX<real_t>>(SrcRec::stas().stla.data(), SrcRec::stas().stla.size());
        Eigen::VectorX<real_t> stlo_vec = Eigen::Map<Eigen::VectorX<real_t>>(SrcRec::stas().stlo.data(), SrcRec::stas().stlo.size());

        lon_min = stlo_vec.minCoeff();
        lon_max = stlo_vec.maxCoeff();
        lat_min = stla_vec.minCoeff();
        lat_max = stla_vec.maxCoeff();
    }

    // Fit parameters of depth-dependent phase function:
    // anomfun(a) = (sqrt(p0^2 + p1*(a-1)) - p0) / p2
    // so that anomfun(anchor_i) ~= n_pi_i, using minpack::lmdif1.
    std::array<real_t, 3> dep_anom(
        const Eigen::VectorX<real_t>& zgrids,
        const int nz,
        const real_t asanom_size
    ) {
        if (nz <= 1) {
            throw std::runtime_error("ModelGrid::dep_anom: nz must be > 1");
        }
        if (zgrids.size() < 2) {
            throw std::runtime_error("ModelGrid::dep_anom: zgrids size must be >= 2");
        }

        auto &logger = ATTLogger::logger();
        const int maxanchor = nz * 2 + 1;
        const real_t dz = zgrids(1) - zgrids(0);
        const real_t nanomtop = asanom_size / dz;
        const real_t anom_size_inc =
            (static_cast<real_t>(zgrids.size() - 1) - static_cast<real_t>(nz) * nanomtop) /
            static_cast<real_t>(2 * nz * (nz - 1));

        Eigen::VectorX<real_t> anch = Eigen::VectorX<real_t>::Zero(maxanchor);
        Eigen::VectorX<real_t> n_pi = Eigen::VectorX<real_t>::Zero(maxanchor);

        anch(0) = _1_CR;
        n_pi(0) = _0_CR;
        logger.Info(
            std::format(
                "Depth anomaly anchor: {:.2f}km, {:.1f}pi",
                (anch(0) - _1_CR) * dz,
                n_pi(0) * static_cast<real_t>(2)
            ),
            MODULE_GRID
        );

        for (int i = 1; i < maxanchor; ++i) {
            anch(i) = anch(i - 1)
                      + (nanomtop - anom_size_inc) / static_cast<real_t>(2)
                      + static_cast<real_t>(i - 1) * anom_size_inc;
            n_pi(i) = n_pi(i - 1) + static_cast<real_t>(0.25);

            logger.Info(
                std::format(
                    "Depth anomaly anchor: {:.2f}km, {:.1f}pi",
                    (anch(i) - _1_CR) * dz,
                    n_pi(i) * static_cast<real_t>(2)
                ),
                MODULE_GRID
            );
        }

        std::vector<double> para = {1.0, 1.0, 1.0};
        std::vector<double> fitfun;
        const double tol = 1.0e-7;

        const auto info = minpack::lmdif1(
            [&anch, &n_pi](int m, int /*n*/, const double* x, double* fvec, int& iflag) {
                constexpr double eps = 1.0e-12;
                if (std::abs(x[2]) < eps) {
                    iflag = -1;
                    return;
                }
                for (int i = 0; i < m; ++i) {
                    const double arg = x[0] * x[0] + x[1] * (static_cast<double>(anch(i)) - 1.0);
                    if (arg < 0.0) {
                        iflag = -1;
                        return;
                    }
                    const double anomfun = (std::sqrt(arg) - x[0]) / x[2];
                    fvec[i] = std::abs(anomfun - static_cast<double>(n_pi(i)));
                }
            },
            maxanchor, 3,para, fitfun, tol
        );

        if (info == minpack::InfoCode::ImproperInput) {
            throw std::runtime_error("ModelGrid::dep_anom: minpack lmdif1 failed with ImproperInput");
        }

        return {
            static_cast<real_t>(para[0]),
            static_cast<real_t>(para[1]),
            static_cast<real_t>(para[2])
        };
    }
}

// Constructor: derive grid dimensions from the station bounding box + margin,
// build evenly-spaced coordinate vectors, and initialise the 3-D model arrays.
ModelGrid::ModelGrid() {
    const auto &dom = InputParams::IP().domain();
    auto &logger = ATTLogger::logger();

    // Grid spacing in lon, lat, and depth directions
    dgrid_i = dom.interval[0];
    dgrid_j = dom.interval[1];
    dgrid_k = dom.interval[2];

    // Expand the station bounding box by num_grid_margin cells on each side
    real_t lon_min, lon_max, lat_min, lat_max;
    get_domain_min_max(lon_min, lon_max, lat_min, lat_max);
    real_t xbeg = lon_min - dom.num_grid_margin * dgrid_i;
    real_t xend = lon_max + dom.num_grid_margin * dgrid_i;
    real_t ybeg = lat_min - dom.num_grid_margin * dgrid_j;
    real_t yend = lat_max + dom.num_grid_margin * dgrid_j;

    // Number of grid nodes (inclusive on both ends)
    n_xyz[0] = static_cast<int>((xend - xbeg) / dgrid_i) + 1;
    n_xyz[1] = static_cast<int>((yend - ybeg) / dgrid_j) + 1;
    n_xyz[2] = static_cast<int>((dom.depth[1] - dom.depth[0]) / dgrid_k) + 1;
    // Expose grid sizes through the I2V macro variables
    ngrid_i = n_xyz[0];
    ngrid_j = n_xyz[1];
    ngrid_k = n_xyz[2];

    xgrids = Eigen::VectorX<real_t>::LinSpaced(n_xyz[0], xbeg, xend);
    ygrids = Eigen::VectorX<real_t>::LinSpaced(n_xyz[1], ybeg, yend);
    zgrids = Eigen::VectorX<real_t>::LinSpaced(n_xyz[2], dom.depth[0], dom.depth[1]);

    logger.Info(
        std::format("Model grids: nx,ny,nz: {}, {}, {},", n_xyz[0], n_xyz[1], n_xyz[2]),
        MODULE_GRID
    );
    logger.Info(
        std::format("Longitude range: {:.3f} to {:.3f}", xbeg, xend),
        MODULE_GRID
    );
    logger.Info(
        std::format("Latitude range: {:.3f} to {:.3f}", ybeg, yend),
        MODULE_GRID
    );
}

// Build a linearly-interpolated 1-D Vs profile spanning [vel_range[0], vel_range[1]]
// at the same depth nodes as zgrids.
void ModelGrid::build_1d_model_linear() {
    vs1d = Eigen::VectorX<real_t>::LinSpaced(
        zgrids.size(), 
        InputParams::IP().inversion().vel_range[0], 
        InputParams::IP().inversion().vel_range[1]
    );
}

// Refine the linear starting model via 1-D surface-wave inversion.
// The linear profile is used as the initial guess.
void ModelGrid::build_1d_model_inversion() {    
    build_1d_model_linear();
    auto inv1d = std::make_unique<Inversion1D>();
    vs1d = inv1d->inv1d(zgrids, vs1d);
}

// Read the "vs" dataset from the HDF5 file specified in init_model_path.
// Validates that the stored grid dimensions match the current model grid.
std::vector<real_t> ModelGrid::load_3d_model() {
    const auto &IP = InputParams::IP();
    std::vector<real_t> model3d;
    H5IO f(IP.inversion().init_model_path, H5IO::RDONLY);
    hsize_t nx = 0, ny = 0, nz = 0;

    try {
        model3d = f.read_volume<real_t>("vs", nx, ny, nz);
        // Ensure the file's grid dimensions are consistent with n_xyz (i,j,k order)
        if (nx != static_cast<hsize_t>(n_xyz[0]) ||
            ny != static_cast<hsize_t>(n_xyz[1]) ||
            nz != static_cast<hsize_t>(n_xyz[2])) {
            throw std::runtime_error("ModelGrid: invalid shape of 3D model in HDF5 file");
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("ModelGrid: failed to load 3D model from HDF5 file: ") + e.what());
    }
    return model3d;
}

// Allocate MPI shared-memory windows for vp3d/vs3d/rho3d, populate the arrays
// according to init_model_type, then broadcast to all ranks.
//
// init_model_type:
//   0 — linear 1-D profile extruded into 3-D
//   1 — 1-D surface-wave inversion result extruded into 3-D
//   2 — full 3-D model loaded directly from an HDF5 file
void ModelGrid::build_init_model() {
    const auto &IP = InputParams::IP();
    auto &mpi = Parallel::mpi();
    auto &logger = ATTLogger::logger();
    
    // Allocate node-local shared memory (all ranks on the same node share these buffers)
    mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], vp3d, win_vp_);
    mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], vs3d, win_vs_);
    mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], rho3d, win_rho_);

    if (IP.inversion().init_model_type == 0) {
        // Build a simple linear Vs gradient
        logger.Info(std::format(
            "Building linear 1-D initial model from {:.2f} to {:.2f}", 
            IP.inversion().vel_range[0], IP.inversion().vel_range[1]), MODULE_GRID
        );
        build_1d_model_linear();
    } else if (IP.inversion().init_model_type == 1) {
        // Invert average dispersion curves to obtain a 1-D Vs model
        logger.Info(
            "Building 1-D initial model from surface-wave inversion of average dispersion curves",
            MODULE_GRID
        );
        build_1d_model_inversion();
    } else if (IP.inversion().init_model_type == 2) {
        // Load an externally-supplied 3-D model; only main rank does the I/O
        logger.Info(std::format(
            "Building 3-D initial model from HDF5 file: {}",
            IP.inversion().init_model_path),
            MODULE_GRID
        );
        if (mpi.is_main()) {
            std::vector<real_t> model3d = load_3d_model();
            std::copy(model3d.begin(), model3d.end(), vs3d);
            // Derive Vp and density from Vs using empirical scaling relations
            vs1d.setZero(ngrid_k);
            for (int k = 0; k < ngrid_k; ++k){
                for (int j = 0; j < ngrid_j; ++j){
                    for (int i = 0; i < ngrid_i; ++i){
                        vs1d(k) += vs3d[I2V(i, j, k)];
                        vp3d[I2V(i, j, k)] = vs2vp(vs3d[I2V(i, j, k)]);
                        rho3d[I2V(i, j, k)] = vp2rho(vp3d[I2V(i, j, k)]);
                    }
                }
                vs1d(k) /= (ngrid_i * ngrid_j);
            }
        }
        mpi.barrier();
    } else {
        throw std::runtime_error("ModelGrid: unknown init_model_type " + std::to_string(IP.inversion().init_model_type));
    }

    if (IP.inversion().init_model_type != 2) {
        // Extrude the 1-D profile laterally to fill the full 3-D volume
        if (mpi.is_main()) {
            for (int ix = 0; ix < n_xyz[0]; ++ix) {
                for (int iy = 0; iy < n_xyz[1]; ++iy) {
                    for (int iz = 0; iz < n_xyz[2]; ++iz) {
                        vp3d[I2V(ix, iy, iz)] = vs2vp(vs1d(iz));
                        vs3d[I2V(ix, iy, iz)] = vs1d(iz);
                        rho3d[I2V(ix, iy, iz)] = vp2rho(vp3d[I2V(ix, iy, iz)]);  // empirical Vp→density scaling
                    }
                }
            }
        }
        mpi.barrier();
    }

    // Broadcast the model from main rank to all other ranks
    mpi.bcast(vs1d.data(), ngrid_k);
    mpi.sync_from_main_rank(vp3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
    mpi.sync_from_main_rank(vs3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
    mpi.sync_from_main_rank(rho3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
}

void ModelGrid::add_perturbation(
    const int nx, const int ny, const int nz,
    const real_t pert_vel, const real_t hmargin,
    const real_t anom_size, const bool only_vs
) {
    auto &mpi = Parallel::mpi();
    const int nelem = ngrid_i * ngrid_j * ngrid_k;

    if (mpi.is_main()) {

        const int ntaper_i = static_cast<int>(hmargin / dgrid_i);
        const int ntaper_j = static_cast<int>(hmargin / dgrid_j);
        const int inner_i = ngrid_i - 2 * ntaper_i;
        const int inner_j = ngrid_j - 2 * ntaper_j;

        if (inner_i <= 0 || inner_j <= 0) {
            throw std::runtime_error(
                "ModelGrid::add_perturbation: hmargin too large for current grid size");
        }

        Eigen::VectorX<real_t> x_pert = Eigen::VectorX<real_t>::Zero(ngrid_i);
        Eigen::VectorX<real_t> y_pert = Eigen::VectorX<real_t>::Zero(ngrid_j);
        Eigen::VectorX<real_t> z_pert = Eigen::VectorX<real_t>::Zero(ngrid_k);

        // x_pert[ntaper_i : ni-ntaper_i-1] = sin(nx*pi*arange(inner_i)/inner_i)
        {
            Eigen::VectorX<real_t> ii = Eigen::VectorX<real_t>::LinSpaced(
                inner_i, _0_CR, static_cast<real_t>(inner_i - 1));
            x_pert.segment(ntaper_i, inner_i) =
                (static_cast<real_t>(nx) * PI * ii.array() / static_cast<real_t>(inner_i)).sin().matrix();
        }

        // y_pert[ntaper_j : nj-ntaper_j-1] = sin(ny*pi*arange(inner_j)/inner_j)
        {
            Eigen::VectorX<real_t> jj = Eigen::VectorX<real_t>::LinSpaced(
                inner_j, _0_CR, static_cast<real_t>(inner_j - 1));
            y_pert.segment(ntaper_j, inner_j) =
                (static_cast<real_t>(ny) * PI * jj.array() / static_cast<real_t>(inner_j)).sin().matrix();
        }

        if (anom_size <= _0_CR) {
            // Uniform-depth wavelength branch.
            Eigen::VectorX<real_t> kk = Eigen::VectorX<real_t>::LinSpaced(
                ngrid_k, _0_CR, static_cast<real_t>(ngrid_k - 1));
            z_pert = (static_cast<real_t>(nz) * PI * kk.array() / static_cast<real_t>(ngrid_k)).sin().matrix();
        } else {
            // Depth-varying wavelength branch fitted by minpack.
            const auto para = dep_anom(zgrids, nz, anom_size);
            for (int k = 0; k < ngrid_k; ++k) {
                const real_t phase = (
                    std::sqrt(para[0] * para[0] + para[1] * static_cast<real_t>(k)) - para[0]
                ) / para[2];
                z_pert(k) = std::sin(static_cast<real_t>(2) * PI * phase);
            }
        }

        // Apply perturbation (vectorized along k for each i,j line)
        for (int i = 0; i < ngrid_i; ++i) {
            for (int j = 0; j < ngrid_j; ++j) {
                const real_t amp = x_pert(i) * y_pert(j) * pert_vel;
                Eigen::Map<Eigen::VectorX<real_t>> vs_line(&vs3d[I2V(i, j, 0)], ngrid_k);
                vs_line.array() *= (_1_CR + amp * z_pert.array());
            }
        }

        if (!only_vs) {
            Eigen::Map<Eigen::VectorX<real_t>> vs_vec(vs3d, nelem);
            Eigen::Map<Eigen::VectorX<real_t>> vp_vec(vp3d, nelem);
            Eigen::Map<Eigen::VectorX<real_t>> rho_vec(rho3d, nelem);
            vp_vec = vs2vp<real_t>(vs_vec);
            rho_vec = vp2rho<real_t>(vp_vec);
        }
    }

    mpi.barrier();
    mpi.sync_from_main_rank(vs3d, nelem);
    mpi.sync_from_main_rank(vp3d, nelem);
    mpi.sync_from_main_rank(rho3d, nelem);
}

void ModelGrid::write(std::string &subname) {
    auto &IP = InputParams::IP();
    auto &mpi = Parallel::mpi();
    if (mpi.is_main()) {
        std::string filename = std::format("{}/{}", IP.output().output_path, subname);
        H5IO f(filename, H5IO::TRUNC);
        f.write_vector("xgrids", xgrids);
        f.write_vector("ygrids", ygrids);
        f.write_vector("zgrids", zgrids);
        f.write_volume("vs", std::vector<real_t>(vs3d, vs3d + ngrid_i * ngrid_j * ngrid_k), ngrid_i, ngrid_j, ngrid_k);
        if (IP.inversion().use_alpha_beta_rho) {
            f.write_volume("vp", std::vector<real_t>(vp3d, vp3d + ngrid_i * ngrid_j * ngrid_k), ngrid_i, ngrid_j, ngrid_k);
            f.write_volume("rho", std::vector<real_t>(rho3d, rho3d + ngrid_i * ngrid_j * ngrid_k), ngrid_i, ngrid_j, ngrid_k);
        }
    }
    mpi.barrier();
}

// Free the MPI shared-memory windows and null the associated raw pointers.
void ModelGrid::release_shm(){
    auto &mpi = Parallel::mpi();

    mpi.free_shared(vp3d, win_vp_);
    mpi.free_shared(vs3d, win_vs_);
    mpi.free_shared(rho3d, win_rho_);
}