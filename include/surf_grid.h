#pragma once

#include "input_params.h"
#include "src_rec.h"
#include "parallel.h"
#include "logger.h"
#include "config.h"
#include "topo.h"
#include "surfker/surfker.hpp"
#include "model_grid.h"
#include "decomposer.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

class SurfGrid {
public:
    // Unified accessor: one SurfGrid per (WaveType, SurfType) combination.
    // Instances are created lazily on first call.
    static SurfGrid& SG(WaveType wt, SurfType vt) {
        static SurfGrid* slots[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
        const int i = static_cast<int>(wt);
        const int j = static_cast<int>(vt);
        if (!slots[i][j]) slots[i][j] = new SurfGrid(wt, vt);
        return *slots[i][j];
    }

    // Backward-compatible aliases (Rayleigh-only call sites)
    static SurfGrid& SG_ph() { return SG(WaveType::RL, SurfType::PH); }
    static SurfGrid& SG_gr() { return SG(WaveType::RL, SurfType::GR); }

    SurfGrid(const SurfGrid&)            = delete;
    SurfGrid& operator=(const SurfGrid&) = delete;
    ~SurfGrid() = default;

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
    Tensor4r sen_vsh_loc, sen_gamma_loc;  // sensitivity kernels for radial anisotropy parameters (if applicable)
    Tensor3r r1_loc, r2_loc;  // anisotropy r1/r2 on local subdomain, shape (loc_nx, loc_ny, nperiod)

    FieldVec ker_loc;
    Tensor3r ker_den_loc;

    void build_media();
    void fwdsurf();
    void compute_dispersion_kernel();
    void correct_depth_with_topo();
    void prepare_aniso_media();

    inline int surf_idx(const int ix, const int iy, const int iper) {
        return ((ix * ngrid_j) + iy) * nperiod_ + iper;
    }
    inline std::string type_name() const { return type_name_; }
    inline int itype() const { return itype_; }
    inline int nperiod() const { return nperiod_; }
    inline WaveType wave_type() const { return wt_; }
    inline int iwave() const { return iwave_of(wt_); }
    inline bool is_active_ker(const int iker) const { return active_kernels_[iker]; }

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

    std::string type_name_;
    WaveType wt_;
    int itype_;
    int nperiod_;
    std::vector<bool> active_kernels_;  // which kernels are active for this SurfGrid (length NPARAMS, indexed by ker_loc)

    SurfGrid(WaveType wt, SurfType vt);
    void release_shm();
    void build_media_matrix_with_topo();
    void setup_active_kernels();
};
