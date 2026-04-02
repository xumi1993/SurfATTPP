#include "model_grid.h"
#include "src_rec.h"
#include "logger.h"
#include "inversion1d.h"
#include "h5io.h"
#include "minpack.hpp"
#include "utils.h"
#include "decomposer.h"
#include "postproc.h"

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
    allocate_model_grids();

    Decomposer::DCP().subdomain_allocation(xgrids, ygrids);
}

void ModelGrid::allocate_model_grids() {
    auto &mpi = Parallel::mpi();
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();

    mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], vp3d, win_vp_);
    mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], vs3d, win_vs_);
    mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], rho3d, win_rho_);
    if (IP.inversion().is_anisotropy) {
        mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], gc3d, win_gc_);
        mpi.alloc_shared(n_xyz[0] * n_xyz[1] * n_xyz[2], gs3d, win_gs_);
    }
    if (run_mode == INVERSION_MODE) {
        vs3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        vp3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        rho3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
        vs3d_loc.setZero();
        vp3d_loc.setZero();
        rho3d_loc.setZero();
        if (IP.inversion().is_anisotropy) {
            // Allocate local sensitivity kernels for anisotropy parameters if needed
            // (not shown here, but would be similar to vs3d_loc/vp3d_loc/rho3d_loc)
            gc3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
            gs3d_loc = Tensor3r(dcp.loc_nx(), dcp.loc_ny(), ngrid_k);
            gc3d_loc.setZero();
            gs3d_loc.setZero();
        }
    }
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

// Read vs (required) and optionally vp, rho, gc, gs from the HDF5 file.
// For any field absent in the file, empirical scaling is applied as fallback.
// Writes directly into the shared-memory arrays vp3d/vs3d/rho3d/gc3d/gs3d.
void ModelGrid::load_3d_model() {
    const auto &IP = InputParams::IP();
    auto &logger = ATTLogger::logger();
    H5IO f(IP.inversion().init_model_path, H5IO::RDONLY);
    hsize_t nx = 0, ny = 0, nz = 0;
    const hsize_t expect_n = static_cast<hsize_t>(n_xyz[0] * n_xyz[1] * n_xyz[2]);

    auto check_dims = [&](const std::string &name) {
        if (nx != static_cast<hsize_t>(n_xyz[0]) ||
            ny != static_cast<hsize_t>(n_xyz[1]) ||
            nz != static_cast<hsize_t>(n_xyz[2])) {
            throw std::runtime_error(std::format(
                "ModelGrid: dataset '{}' shape ({},{},{}) != grid ({},{},{})",
                name, nx, ny, nz, n_xyz[0], n_xyz[1], n_xyz[2]));
        }
    };

    try {
        // --- vs (required) ---
        {
            auto d = f.read_volume<real_t>("vs", nx, ny, nz);
            check_dims("vs");
            std::copy(d.begin(), d.end(), vs3d);
            logger.Info("Loaded 'vs' from HDF5 file.", MODULE_GRID);
        }

        // --- vp (optional, fallback: empirical vs2vp) ---
        if (f.exists("vp")) {
            auto d = f.read_volume<real_t>("vp", nx, ny, nz);
            check_dims("vp");
            std::copy(d.begin(), d.end(), vp3d);
            logger.Info("Loaded 'vp' from HDF5 file.", MODULE_GRID);
        } else {
            logger.Info("'vp' not found in HDF5 file, computing from empirical vs2vp.", MODULE_GRID);
            for (hsize_t i = 0; i < expect_n; ++i)
                vp3d[i] = vs2vp(vs3d[i]);
        }

        // --- rho (optional, fallback: empirical vp2rho) ---
        if (f.exists("rho")) {
            auto d = f.read_volume<real_t>("rho", nx, ny, nz);
            check_dims("rho");
            std::copy(d.begin(), d.end(), rho3d);
            logger.Info("Loaded 'rho' from HDF5 file.", MODULE_GRID);
        } else {
            logger.Info("'rho' not found in HDF5 file, computing from empirical vp2rho.", MODULE_GRID);
            for (hsize_t i = 0; i < expect_n; ++i)
                rho3d[i] = vp2rho(vp3d[i]);
        }

        // --- gc / gs (optional, only if is_anisotropy) ---
        if (IP.inversion().is_anisotropy) {
            if (f.exists("gc")) {
                auto d = f.read_volume<real_t>("gc", nx, ny, nz);
                check_dims("gc");
                std::copy(d.begin(), d.end(), gc3d);
                logger.Info("Loaded 'gc' from HDF5 file.", MODULE_GRID);
            } else {
                logger.Info("'gc' not found in HDF5 file, initialising to zero.", MODULE_GRID);
                std::fill(gc3d, gc3d + expect_n, _0_CR);
            }
            if (f.exists("gs")) {
                auto d = f.read_volume<real_t>("gs", nx, ny, nz);
                check_dims("gs");
                std::copy(d.begin(), d.end(), gs3d);
                logger.Info("Loaded 'gs' from HDF5 file.", MODULE_GRID);
            } else {
                logger.Info("'gs' not found in HDF5 file, initialising to zero.", MODULE_GRID);
                std::fill(gs3d, gs3d + expect_n, _0_CR);
            }
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(
            std::string("ModelGrid: failed to load 3D model from HDF5 file: ") + e.what());
    }

    // Compute depth-averaged vs1d from the loaded vs3d
    vs1d.setZero(ngrid_k);
    for (int k = 0; k < ngrid_k; ++k) {
        for (int j = 0; j < ngrid_j; ++j)
            for (int i = 0; i < ngrid_i; ++i)
                vs1d(k) += vs3d[I2V(i, j, k)];
        vs1d(k) /= (ngrid_i * ngrid_j);
    }
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

    if (IP.inversion().init_model_type == 0) {
        // Build a simple linear Vs gradient
        logger.Info(std::format(
            "Building linear 1-D initial model with S-wave velocity from {:.2f} to {:.2f} km/s", 
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
            load_3d_model();  // fills vs3d/vp3d/rho3d and optionally gc3d/gs3d
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
                        if (IP.inversion().is_anisotropy) {
                            gc3d[I2V(ix, iy, iz)] = _0_CR;  // start with isotropic model
                            gs3d[I2V(ix, iy, iz)] = _0_CR;  // start with isotropic model
                        }
                    }
                }
            }
            if (IP.output().output_initial_model) {
                write(std::string("initial_model.h5"));
            }
        }
        mpi.barrier();
    }

    // Broadcast the model from main rank to all other ranks
    mpi.bcast(vs1d.data(), ngrid_k);
    mpi.sync_from_main_rank(vp3d,  n_xyz[0] * n_xyz[1] * n_xyz[2]);
    mpi.sync_from_main_rank(vs3d,  n_xyz[0] * n_xyz[1] * n_xyz[2]);
    mpi.sync_from_main_rank(rho3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
    if (IP.inversion().is_anisotropy) {
        mpi.sync_from_main_rank(gc3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
        mpi.sync_from_main_rank(gs3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
    }
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

void ModelGrid::collect_model_loc() {
    auto &mpi = Parallel::mpi();
    auto &dcp = Decomposer::DCP();
    auto &IP = InputParams::IP();

    int nelem = ngrid_i * ngrid_j * ngrid_k;
    Tensor3r vs_tmp = dcp.collect_data(vs3d_loc.data());
    Tensor3r vp_tmp = dcp.collect_data(vp3d_loc.data());
    Tensor3r rho_tmp = dcp.collect_data(rho3d_loc.data());
    Tensor3r gc_tmp, gs_tmp;
    if (IP.inversion().is_anisotropy) {
        gc_tmp = dcp.collect_data(gc3d_loc.data());
        gs_tmp = dcp.collect_data(gs3d_loc.data());
    }
    if (mpi.is_main()) {
        std::copy(vs_tmp.data(), vs_tmp.data() + nelem, vs3d);
        std::copy(vp_tmp.data(), vp_tmp.data() + nelem, vp3d);
        std::copy(rho_tmp.data(), rho_tmp.data() + nelem, rho3d);
         if (IP.inversion().is_anisotropy) {
            std::copy(gc_tmp.data(), gc_tmp.data() + nelem, gc3d);
            std::copy(gs_tmp.data(), gs_tmp.data() + nelem, gs3d);
        }
    }
    mpi.barrier();
    mpi.sync_from_main_rank(vs3d, nelem);
    mpi.sync_from_main_rank(vp3d, nelem);
    mpi.sync_from_main_rank(rho3d, nelem);
    if (IP.inversion().is_anisotropy) {
        mpi.sync_from_main_rank(gc3d, nelem);
        mpi.sync_from_main_rank(gs3d, nelem);
    }
   
}

void ModelGrid::write(const std::string &subname) {
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
