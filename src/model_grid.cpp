#include "model_grid.h"
#include "src_rec.h"
#include "logger.h"
#include "inversion1d.h"
#include "h5io.h"
#include "utils.h"


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
    std::vector<real_t> model3d, model_tmp;
    H5IO f(IP.inversion().init_model_path, H5IO::RDONLY);
    hsize_t nz, ny, nx;

    try {
        model_tmp = f.read_volume<real_t>("vs", nz, ny, nx);
        // Ensure the file's grid dimensions are consistent with n_xyz
        if (nx != static_cast<hsize_t>(n_xyz[0]) || ny != static_cast<hsize_t>(n_xyz[1]) || nz != static_cast<hsize_t>(n_xyz[2])) {
            throw std::runtime_error("ModelGrid: invalid shape of 3D model in HDF5 file");
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("ModelGrid: failed to load 3D model from HDF5 file: ") + e.what());
    }
    model3d.resize(nz * ny * nx);
    for (int k = 0; k < ngrid_k; ++k) {
        for (int j = 0; j < ngrid_j; ++j) {
            for (int i = 0; i < ngrid_i; ++i) {
                const int isrc = k * ngrid_j * ngrid_i + j * ngrid_i + i;
                model3d[I2V(i, j, k)] = model_tmp[isrc];
            }
        }
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
    mpi.sync_from_main_rank(vp3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
    mpi.sync_from_main_rank(vs3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
    mpi.sync_from_main_rank(rho3d, n_xyz[0] * n_xyz[1] * n_xyz[2]);
}

// Free the MPI shared-memory windows and null the associated raw pointers.
void ModelGrid::release_shm(){
    auto &mpi = Parallel::mpi();

    mpi.free_shared(vp3d, win_vp_);
    mpi.free_shared(vs3d, win_vs_);
    mpi.free_shared(rho3d, win_rho_);
}