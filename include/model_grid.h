#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "logger.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

class ModelGrid {
public:
    static void init() {
        get_instance_ptr() = std::make_unique<ModelGrid>();
    }
    static ModelGrid& MG() {
        auto* ptr = get_instance_ptr().get();
        if (!ptr) throw std::runtime_error("ModelGrid: call init() first");
        return *ptr;
    }

    ModelGrid();
    ~ModelGrid() { release_shm(); }

    void build_init_model();
    void add_perturbation(
        const int nx, const int ny, const int nz,
        const real_t pert_vel, const real_t hmargin = _0_CR,
        const real_t anom_size = _0_CR, const bool only_vs = false
    );
    void add_aniso_perturbation(
        const int nx, const int ny, const int nz,
        const real_t pert_gc, const real_t pert_gs,
        const real_t hmargin = _0_CR, const real_t anom_size = _0_CR
    );
    void write(const std::string &subname);
    void collect_model_loc();

    Eigen::VectorX<real_t> vs1d;  // 1D S-wave velocity model for depth kernel calculation
    Eigen::VectorX<real_t> xgrids;
    Eigen::VectorX<real_t> ygrids;
    Eigen::VectorX<real_t> zgrids;

    real_t* vp3d;
    real_t* vs3d;
    real_t* vsh3d;
    real_t* rho3d;
    real_t* gc3d;
    real_t* gs3d;

    Tensor3r vs3d_loc;  // local subdomain of vs3d for each rank, with halo regions included
    Tensor3r vp3d_loc;  // local subdomain of vp3d for each rank, with halo regions included
    Tensor3r rho3d_loc;  // local subdomain of rho3d for each rank, with halo regions included
    Tensor3r gc3d_loc;  // local anisotropy Gc parameter (cosine component), shape (loc_nx, loc_ny, ngrid_k)
    Tensor3r gs3d_loc;  // local anisotropy Gs parameter (sine component), shape (loc_nx, loc_ny, ngrid_k)
    Tensor3r vsh3d_loc;  // local subdomain of vsh3d for each rank, with halo regions included (for azimuthal anisotropy only)
    Tensor3r gamma_loc;  // local subdomain of gamma (for radial anisotropy only)

private:
    static std::unique_ptr<ModelGrid> &get_instance_ptr() {
        static std::unique_ptr<ModelGrid> MG;
        return MG;
    }

    MPI_Win win_vp_  = MPI_WIN_NULL;
    MPI_Win win_vs_  = MPI_WIN_NULL;
    MPI_Win win_rho_ = MPI_WIN_NULL;
    MPI_Win win_gc_  = MPI_WIN_NULL;
    MPI_Win win_gs_  = MPI_WIN_NULL;

    void build_1d_model_linear();
    void build_1d_model_inversion();
    void load_3d_model();
    void release_shm();
    void allocate_model_grids();

    // Build separable sinusoidal spatial pattern vectors (x_p, y_p, z_p) for
    // checkerboard perturbations. hmargin controls the taper-free edge width.
    std::tuple<Eigen::VectorX<real_t>, Eigen::VectorX<real_t>, Eigen::VectorX<real_t>>
    build_perturbation_pattern(int nx_w, int ny_w, int nz_w,
                               real_t hmargin, real_t anom_sz) const;
};
