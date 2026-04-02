#pragma once

#include "input_params.h"
#include "src_rec.h"
#include "parallel.h"
#include "logger.h"
#include "config.h"
#include "topo.h"
#include "surfdisp.h"
#include "model_grid.h"
#include "decomposer.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

class SurfGrid {
public:
    // Phase-velocity grid
    static SurfGrid& SG_ph() {
        static SurfGrid inst(surfType::PH);
        return inst;
    }

    // Group-velocity grid
    static SurfGrid& SG_gr() {
        static SurfGrid inst(surfType::GR);
        return inst;
    }

    SurfGrid(const SurfGrid&)            = delete;
    SurfGrid& operator=(const SurfGrid&) = delete;
    ~SurfGrid() = default;

    int nperiod;
    int itype;  // 0: phase velocity, 1: group velocity
    real_t* svel;  // length nperiod, phase or group velocity at each period
    real_t* a;
    real_t* b;
    real_t* c;
    real_t* topo_angle;
    real_t* m11;
    real_t* m12;
    real_t* m22;
    real_t* ref_t;

    std::vector<Eigen::MatrixX<real_t>> adj_s_local;  // length nperiod, each is a matrix of svel on the surface grid
    std::vector<Eigen::MatrixX<real_t>> adj_xi_local;  // length nperiod, each is a matrix of xi on the surface grid
    std::vector<Eigen::MatrixX<real_t>> adj_eta_local;  // length nperiod, each is a matrix of eta on the surface grid
    std::vector<Eigen::MatrixX<real_t>> kden_s_local;    // length nperiod, each is a vector of a on the surface grid
    
    Tensor4r sen_vp_loc, sen_vs_loc, sen_rho_loc;  // sensitivity kernels for vp, vs, rho with shape (ngrid_i, ngrid_j, ngrid_k, nperiod)
    Tensor4r sen_gc_loc, sen_gs_loc;  // sensitivity kernels for anisotropy parameters (if applicable)
    Tensor3r r1_loc, r2_loc;  // anisotropy r1/r2 on local subdomain, shape (loc_nx, loc_ny, nperiod)

    std::vector<Tensor3r> ker_loc;
    Tensor3r ker_den_loc;

    void build_media();
    void fwdsurf();
    void compute_dispersion_kernel();
    void correct_depth_with_topo();
    void prepare_aniso_media();

    inline int surf_idx(const int ix, const int iy, const int iper) {
        return ((ix * ngrid_j) + iy) * nperiod + iper;
    }


private:
    MPI_Win win_svel_ = MPI_WIN_NULL;
    MPI_Win win_a_    = MPI_WIN_NULL;
    MPI_Win win_b_    = MPI_WIN_NULL;
    MPI_Win win_c_    = MPI_WIN_NULL;
    MPI_Win win_topo_ = MPI_WIN_NULL;
    MPI_Win win_m11_  = MPI_WIN_NULL;
    MPI_Win win_m12_  = MPI_WIN_NULL;
    MPI_Win win_m22_  = MPI_WIN_NULL;
    MPI_Win win_ref_t_ = MPI_WIN_NULL;

    explicit SurfGrid(surfType tp);
    void release_shm();
    void build_media_matrix_with_topo();
};
