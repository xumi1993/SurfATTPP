#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "logger.h"
#include <Eigen/Core>
#include <Eigen/Dense>
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
    void write(const std::string &subname);

    Eigen::VectorX<real_t> vs1d;  // 1D S-wave velocity model for depth kernel calculation
    std::vector<int> n_xyz = {0, 0, 0};  // number of grid points in x/y/z directions (excluding margins)
    Eigen::VectorX<real_t> xgrids;
    Eigen::VectorX<real_t> ygrids;
    Eigen::VectorX<real_t> zgrids;

    real_t* vp3d;
    real_t* vs3d;
    real_t* rho3d;

private:
    static std::unique_ptr<ModelGrid> &get_instance_ptr() {
        static std::unique_ptr<ModelGrid> MG;
        return MG;
    }

    MPI_Win win_vp_ = MPI_WIN_NULL;
    MPI_Win win_vs_ = MPI_WIN_NULL;
    MPI_Win win_rho_ = MPI_WIN_NULL;

    void build_1d_model_linear();
    void build_1d_model_inversion();
    std::vector<real_t> load_3d_model();
    void release_shm();
};
