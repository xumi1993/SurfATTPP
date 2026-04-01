#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "model_grid.h"
#include "surf_grid.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <stdexcept>
#include <vector>

class PostProc {
public:
    // -----------------------------------------------------------------------
    // Singleton access — lazily constructed on first call to PP().
    //   Requires ModelGrid and InputParams to be ready beforehand.
    //   auto &pp = PostProc::PP();
    // -----------------------------------------------------------------------
    static PostProc &PP() {
        static PostProc inst;
        return inst;
    }

    PostProc();

    // -----------------------------------------------------------------------
    // InvGrid — inversion sub-grid (coarser than ModelGrid)
    // -----------------------------------------------------------------------
    struct InvGrid {
        int                       nset = 1;  // number of staggered sets
        // Each column is one staggered set: shape (nx, nset), (ny, nset), (nz, nset)
        Eigen::MatrixX<real_t>    xinv;
        Eigen::MatrixX<real_t>    yinv;
        Eigen::MatrixX<real_t>    zinv;

        int n_inv_I, n_inv_J, n_inv_K;

        // Match Fortran construct_inv_grids:
        //   n_inv : {ninvx, ninvy, ninvz}  (0 → use ModelGrid size)
        //   nset_ : number of staggered sets (n_inv_components)
        void init(const std::vector<int> &n_inv, int nset_);
        std::vector<real_t> fwd2inv(const Eigen::Tensor<real_t, 3, Eigen::RowMajor> buf);
        Eigen::Tensor<real_t, 3, Eigen::RowMajor> inv2fwd(const real_t *buf);
        inline int I2V_INV_GRIDS(const int A, const int B, const int C, const int D) {
            return (A*n_inv_K*n_inv_J*nset + B*n_inv_K*nset + C*nset + D);  // 4D to 1D index
        }
    };

    Eigen::Tensor<real_t, 3, Eigen::RowMajor> smooth(const Eigen::Tensor<real_t, 3, Eigen::RowMajor> buf);
    InvGrid inv_grid;      // isotropic inversion grid
    InvGrid inv_grid_ani;  // anisotropy inversion grid

private:
    PostProc(const PostProc &)            = delete;
    PostProc &operator=(const PostProc &) = delete;
    Eigen::Tensor<real_t, 3, Eigen::RowMajor> pde_smooth(const Eigen::Tensor<real_t, 3, Eigen::RowMajor> buf);
   

};