#pragma once

#include "input_params.h"
#include "parallel.h"
#include "logger.h"
#include "utils.h"
#include "config.h"

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


class Decomposer {
public:
    static Decomposer& DCP();

    void subdomain_allocation(
        const Eigen::VectorX<real_t> &xgrids, 
        const Eigen::VectorX<real_t> &ygrids
    );
    Decomposer(const Decomposer&) = delete;
    Decomposer& operator=(const Decomposer&) = delete;
    void prepare_expanded_field(real_t* arr);
    Tensor3r collect_data(real_t* buf_loc);
    Tensor3r distribute_data(real_t* buf);

    int rank()              const { return rank_; }
    int size()              const { return size_; }
    int loc_I_start()       const { return loc_I_start_; }
    int loc_I_end()         const { return loc_I_end_; }
    int loc_J_start()       const { return loc_J_start_; }
    int loc_J_end()         const { return loc_J_end_; }
    int loc_nx()            const { return loc_nx_; }
    int loc_ny()            const { return loc_ny_; }
    int loc_nx_expd()       const { return loc_nx_expd_; }
    int loc_ny_expd()       const { return loc_ny_expd_; }
    std::vector<int> neighbors_id() const { return neighbors_id_; }
    std::vector<std::vector<int>> glob_I;
    std::vector<std::vector<int>> glob_J;
    Tensor3r expd_field;
    Eigen::VectorX<real_t> x_loc_expd, y_loc_expd;

private:
    Decomposer();

    void decompose_impl();

    std::pair<int, int> close_factors(int nx, int ny, int num);
    void prepare_boundary_data_to_send(real_t* arr);
    void send_recv_boundary_data(real_t* arr);

    int rank_ = 0;
    int size_ = 1;
    int loc_I_start_ = 0;
    int loc_I_end_ = 0;
    int loc_J_start_ = 0;
    int loc_J_end_ = 0;
    int ndiv_x_ = 1;
    int ndiv_y_ = 1;
    int loc_nx_ = 0;
    int loc_ny_ = 0;
    int loc_nx_expd_ = 0;
    int loc_ny_expd_ = 0;
    std::vector<int> neighbors_id_ = {-1, -1, -1, -1, -1, -1, -1, -1};
    std::vector<std::vector<real_t>> bound_dat_send = std::vector<std::vector<real_t>>(8);
    std::vector<std::vector<real_t>> bound_dat_recv = std::vector<std::vector<real_t>>(8);
    std::vector<MPI_Request> mpi_send_reqs;
    std::vector<MPI_Request> mpi_recv_reqs;

    inline int I2V_bound(const int A, const int B){
        return (A)*ngrid_k + (B);
    }

    inline int I2V_loc(const int A, const int B, const int C){
        return ((A)*loc_ny_*ngrid_k + (B)*ngrid_k + (C));
    }

};
