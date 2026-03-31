#include "decomposer.h"
#include "model_grid.h"

#include <cmath>


Decomposer& Decomposer::DCP() {
    static Decomposer inst;
    return inst;
}

Decomposer::Decomposer() {
    auto& mpi = Parallel::mpi();
    rank_ = mpi.rank();
    size_ = mpi.size();

    decompose_impl();
}

void Decomposer::decompose_impl() {
    auto &mpi = Parallel::mpi();

    std::tie(ndiv_x_, ndiv_y_) = close_factors(
        ngrid_i, ngrid_j, size_);
    
    loc_I_start_ = (rank_ % ndiv_x_) * (ngrid_i / ndiv_x_);
    loc_I_end_ = loc_I_start_ + (ngrid_i / ndiv_x_) - 1;
    loc_J_start_ =  (rank_ / ndiv_x_) * (ngrid_j / ndiv_y_);
    loc_J_end_ = loc_J_start_ + (ngrid_j / ndiv_y_) - 1;
    
    int rank_v, max_rank_v;
    if ( ngrid_i % ndiv_x_ != 0 ) {
        rank_v = rank_ % ndiv_x_;
        mpi.max_all_all(rank_v, max_rank_v);
        if (rank_v == max_rank_v) {
            loc_I_end_ = ngrid_i - 1;
        }
    }
    if ( ngrid_j % ndiv_y_ != 0 ) {
        rank_v = rank_ / ndiv_x_;
        mpi.max_all_all(rank_v, max_rank_v);
        if (rank_v == max_rank_v) {
            loc_J_end_ = ngrid_j - 1;
        }
    }

    std::vector<std::vector<int>> loc_ix(size_, std::vector<int>(2));
    std::vector<std::vector<int>> loc_iy(size_, std::vector<int>(2));
    
    loc_ix[rank_][0] = loc_I_start_;
    loc_ix[rank_][1] = loc_I_end_;
    loc_nx_ = loc_I_end_ - loc_I_start_ + 1;
    loc_iy[rank_][0] = loc_J_start_;
    loc_iy[rank_][1] = loc_J_end_;
    loc_ny_ = loc_J_end_ - loc_J_start_ + 1;

    glob_I.resize(size_, std::vector<int>(2));
    glob_J.resize(size_, std::vector<int>(2));
    for (int i = 0; i < size_; i++) {
        mpi.sum_all_all(loc_ix[i].data(), glob_I[i].data(), 2);
        mpi.sum_all_all(loc_iy[i].data(), glob_J[i].data(), 2);
    }

    // get neighbor rank id
    neighbors_id_[0] = (rank_ % ndiv_x_ == 0) ? -1 : rank_ - 1;                   // -x neighbor
    neighbors_id_[1] = (rank_ % ndiv_x_ == ndiv_x_ - 1) ? -1 : rank_ + 1;         // +x neighbor
    neighbors_id_[2] = (rank_ / ndiv_x_ == 0) ? -1 : rank_ - ndiv_x_;             // -y neighbor
    neighbors_id_[3] = (rank_ / ndiv_x_ == ndiv_y_ - 1) ? -1 : rank_ + ndiv_x_;   // +y neighbor
    // get left-bottom neighbor rank id
    neighbors_id_[4] = (neighbors_id_[0] == -1 || neighbors_id_[2] == -1) ? -1 : neighbors_id_[2] - 1;
    // get right-bottom neighbor rank id
    neighbors_id_[5] = (neighbors_id_[1] == -1 || neighbors_id_[2] == -1) ? -1 : neighbors_id_[2] + 1;
    // get left-top neighbor rank id
    neighbors_id_[6] = (neighbors_id_[0] == -1 || neighbors_id_[3] == -1) ? -1 : neighbors_id_[3] - 1;
    // get right-top neighbor rank id
    neighbors_id_[7] = (neighbors_id_[1] == -1 || neighbors_id_[3] == -1) ? -1 : neighbors_id_[3] + 1;

    loc_nx_expd_ = loc_nx_;
    loc_ny_expd_ = loc_ny_;

    if (neighbors_id_[0] != -1) {
        loc_nx_expd_ += 1;
    }

    if (neighbors_id_[1] != -1) {
        loc_nx_expd_ += 1;
    }

    if (neighbors_id_[2] != -1) {
        loc_ny_expd_ += 1;
    }

    if (neighbors_id_[3] != -1) {
        loc_ny_expd_ += 1;
    }
    mpi.barrier();
}

void Decomposer::subdomain_allocation() {
    auto& mg = ModelGrid::MG();
    auto& mpi = Parallel::mpi();

    expd_field = Eigen::Tensor<real_t, 3, Eigen::RowMajor>(loc_nx_expd_, loc_ny_expd_, ngrid_k);
    expd_field.setZero();
    x_loc_expd = Eigen::VectorX<real_t>::Zero(loc_nx_expd_);
    y_loc_expd = Eigen::VectorX<real_t>::Zero(loc_ny_expd_);
    int i_shift = (neighbors_id_[0] != -1) ? -1 : 0;
    int j_shift = (neighbors_id_[2] != -1) ? -1 : 0;
    for (int i = 0; i < loc_nx_expd_; i++) {
        x_loc_expd(i) = mg.xgrids(loc_I_start_) + (i + i_shift) * dgrid_i;
    }

    for (int j = 0; j < loc_ny_expd_; j++) {
        y_loc_expd(j) = mg.ygrids(loc_J_start_) + (j + j_shift) * dgrid_j;
    }

    // allocate memory for MPI requests
    mpi_send_reqs.resize(8);
    mpi_recv_reqs.resize(8);
    for (int i = 0; i < 8; i++){
        mpi_send_reqs[i] = MPI_REQUEST_NULL;
        mpi_recv_reqs[i] = MPI_REQUEST_NULL;
    }

    // allocate memory for boundary data
     // left boundary
    bound_dat_send[0].resize(loc_ny_ * ngrid_k);
    bound_dat_recv[0].resize(loc_ny_ * ngrid_k);
    // right boundary
    bound_dat_send[1].resize(loc_ny_ * ngrid_k);
    bound_dat_recv[1].resize(loc_ny_ * ngrid_k);
    // bottom boundary
    bound_dat_send[2].resize(loc_nx_ * ngrid_k);
    bound_dat_recv[2].resize(loc_nx_ * ngrid_k);
    // top boundary
    bound_dat_send[3].resize(loc_nx_ * ngrid_k);
    bound_dat_recv[3].resize(loc_nx_ * ngrid_k);
    // left-bottom corner
    bound_dat_send[4].resize(ngrid_k);
    bound_dat_recv[4].resize(ngrid_k);
    // right-bottom corner
    bound_dat_send[5].resize(ngrid_k);
    bound_dat_recv[5].resize(ngrid_k);
    // left-top corner
    bound_dat_send[6].resize(ngrid_k);
    bound_dat_recv[6].resize(ngrid_k);
    // right-top corner
    bound_dat_send[7].resize(ngrid_k);
    bound_dat_recv[7].resize(ngrid_k);

    mpi.barrier();
}

void Decomposer::prepare_boundary_data_to_send(real_t* arr) {
// store the pointers to the ghost layer's elements to be sent / to receive

    // left boundary
    for (int j = 0; j < loc_ny_; j++) {
        for (int k = 0; k < ngrid_k; k++) {
            bound_dat_send[0][I2V_bound(j,k)] = arr[I2V_loc(0, j, k)];
        }
    }

    // right boundary
    for (int j = 0; j < loc_ny_; j++) {
        for (int k = 0; k < ngrid_k; k++) {
            bound_dat_send[1][I2V_bound(j,k)] = arr[I2V_loc(loc_nx_-1, j, k)];
        }
    }

    // bottom boundary
    for (int i = 0; i < loc_nx_; i++) {
        for (int k = 0; k < ngrid_k; k++) {
            bound_dat_send[2][I2V_bound(i,k)] = arr[I2V_loc(i, 0, k)];
        }
    }

    // top boundary
    for (int i = 0; i < loc_nx_; i++) {
        for (int k = 0; k < ngrid_k; k++) {
            bound_dat_send[3][I2V_bound(i,k)] = arr[I2V_loc(i, loc_ny_-1, k)];
        }
    }

    // left-bottom corner
    for (int k = 0; k < ngrid_k; k++) {
        bound_dat_send[4][k] = arr[I2V_loc(0, 0, k)];
    }

    // right-bottom corner
    for (int k = 0; k < ngrid_k; k++) {
        bound_dat_send[5][k] = arr[I2V_loc(loc_nx_-1, 0, k)];
    }

    // left-top corner
    for (int k = 0; k < ngrid_k; k++) {
        bound_dat_send[6][k] = arr[I2V_loc(0, loc_ny_-1, k)];
    }

    // right-top corner
    for (int k = 0; k < ngrid_k; k++) {
        bound_dat_send[7][k] = arr[I2V_loc(loc_nx_-1, loc_ny_-1, k)];
    }

}

void Decomposer::send_recv_boundary_data(real_t* arr) {
    auto& mpi = Parallel::mpi();
    
    prepare_boundary_data_to_send(arr);

    if (neighbors_id_[0] != -1) {
        mpi.isend(bound_dat_send[0].data(), loc_ny_ * ngrid_k, neighbors_id_[0], MPI_TAG_BASE, mpi_send_reqs[0]);
        mpi.irecv(bound_dat_recv[0].data(), loc_ny_ * ngrid_k, neighbors_id_[0], MPI_TAG_BASE, mpi_recv_reqs[0]);
    }

    if (neighbors_id_[1] != -1) {
        mpi.isend(bound_dat_send[1].data(), loc_ny_ * ngrid_k, neighbors_id_[1], MPI_TAG_BASE, mpi_send_reqs[1]);
        mpi.irecv(bound_dat_recv[1].data(), loc_ny_ * ngrid_k, neighbors_id_[1], MPI_TAG_BASE, mpi_recv_reqs[1]);
    }

    if (neighbors_id_[2] != -1) {
        mpi.isend(bound_dat_send[2].data(), loc_nx_ * ngrid_k, neighbors_id_[2], MPI_TAG_BASE, mpi_send_reqs[2]);
        mpi.irecv(bound_dat_recv[2].data(), loc_nx_ * ngrid_k, neighbors_id_[2], MPI_TAG_BASE, mpi_recv_reqs[2]);
    }

    if (neighbors_id_[3] != -1) {
        mpi.isend(bound_dat_send[3].data(), loc_nx_ * ngrid_k, neighbors_id_[3], MPI_TAG_BASE, mpi_send_reqs[3]);
        mpi.irecv(bound_dat_recv[3].data(), loc_nx_ * ngrid_k, neighbors_id_[3], MPI_TAG_BASE, mpi_recv_reqs[3]);
    }

    if (neighbors_id_[4] != -1) {
        mpi.isend(bound_dat_send[4].data(), ngrid_k, neighbors_id_[4], MPI_TAG_BASE, mpi_send_reqs[4]);
        mpi.irecv(bound_dat_recv[4].data(), ngrid_k, neighbors_id_[4], MPI_TAG_BASE, mpi_recv_reqs[4]);
    }

    if (neighbors_id_[5] != -1) {
        mpi.isend(bound_dat_send[5].data(), ngrid_k, neighbors_id_[5], MPI_TAG_BASE, mpi_send_reqs[5]);
        mpi.irecv(bound_dat_recv[5].data(), ngrid_k, neighbors_id_[5], MPI_TAG_BASE, mpi_recv_reqs[5]);
    }

    if (neighbors_id_[6] != -1) {
        mpi.isend(bound_dat_send[6].data(), ngrid_k, neighbors_id_[6], MPI_TAG_BASE, mpi_send_reqs[6]);
        mpi.irecv(bound_dat_recv[6].data(), ngrid_k, neighbors_id_[6], MPI_TAG_BASE, mpi_recv_reqs[6]);
    }

    if (neighbors_id_[7] != -1) {
        mpi.isend(bound_dat_send[7].data(), ngrid_k, neighbors_id_[7], MPI_TAG_BASE, mpi_send_reqs[7]);
        mpi.irecv(bound_dat_recv[7].data(), ngrid_k, neighbors_id_[7], MPI_TAG_BASE, mpi_recv_reqs[7]);
    }

    // wait for finishing communication
    for (int i = 0; i < 8; i++) {
        if (neighbors_id_[i] != -1) {
            mpi.wait_req(mpi_send_reqs[i]);
            mpi.wait_req(mpi_recv_reqs[i]);
        }
    }
    mpi.barrier();

}

void Decomposer::prepare_expanded_field(real_t* arr) {
    send_recv_boundary_data(arr);
    expd_field.setZero();

    int ix_start, iy_start;

    ix_start = (neighbors_id_[0] != -1) ? 1 : 0;
    iy_start = (neighbors_id_[2] != -1) ? 1 : 0;

    // copy local field to the expanded field
    for (int i = 0; i < loc_nx_; i++) {
        for (int j = 0; j < loc_ny_; j++) {
            for (int k = 0; k < ngrid_k; k++) {
                expd_field(i+ix_start, j+iy_start, k) = arr[I2V_loc(i, j, k)];
            }
        }
    }

    // copy received boundary data to the expanded field
    // left boundary
    if (neighbors_id_[0] != -1) {
        for (int j = 0; j < loc_ny_; j++) {
            for (int k = 0; k < ngrid_k; k++) {
                expd_field(0, j+iy_start, k) = bound_dat_recv[0][I2V_bound(j,k)];
            }
        }
    }

    // right boundary
    if (neighbors_id_[1] != -1) {
        for (int j = 0; j < loc_ny_; j++) {
            for (int k = 0; k < ngrid_k; k++) {
                expd_field(loc_nx_expd_-1, j+iy_start, k) = bound_dat_recv[1][I2V_bound(j,k)];
            }
        }
    }

    // bottom boundary
    if (neighbors_id_[2] != -1) {
        for (int i = 0; i < loc_nx_; i++) {
            for (int k = 0; k < ngrid_k; k++) {
                expd_field(i+ix_start, 0, k) = bound_dat_recv[2][I2V_bound(i,k)];
            }   
        }
    }

    // top boundary
    if (neighbors_id_[3] != -1) {
        for (int i = 0; i < loc_nx_; i++) {
            for (int k = 0; k < ngrid_k; k++) {
                expd_field(i+ix_start, loc_ny_expd_-1, k) = bound_dat_recv[3][I2V_bound(i,k)];
            }
        }
    }

    // left-bottom corner
    if (neighbors_id_[4] != -1) {
        for (int k = 0; k < ngrid_k; k++) {
            expd_field(0, 0, k) = bound_dat_recv[4][k];
        }
    }

    // right-bottom corner
    if (neighbors_id_[5] != -1) {
        for (int k = 0; k < ngrid_k; k++) {
            expd_field(loc_nx_expd_-1, 0, k) = bound_dat_recv[5][k];
        }
    }

    // left-top corner
    if (neighbors_id_[6] != -1) {
        for (int k = 0; k < ngrid_k; k++) {
            expd_field(0, loc_ny_expd_-1, k) = bound_dat_recv[6][k];
        }
    }

    // right-top corner
    if (neighbors_id_[7] != -1) {
        for (int k = 0; k < ngrid_k; k++) {
            expd_field(loc_nx_expd_-1, loc_ny_expd_-1, k) = bound_dat_recv[7][k];
        }
    }

}

std::pair<int, int> Decomposer::close_factors(int nx, int ny, int num) {
    int f1o, f2o;
    if (num <= 0 || ny == 0) {
        f1o = 1;
        f2o = (num > 0) ? num : 1;
        return {f1o, f2o};
    }

    const real_t dif0 = static_cast<real_t>(nx) / static_cast<real_t>(ny);

    int f1 = 1;
    int f2 = num;
    f1o = f1;
    f2o = f2;

    real_t dd = std::abs(dif0 - static_cast<real_t>(f1) / static_cast<real_t>(f2));

    for (int i = 2; i <= num; ++i) {
        if (num % i == 0) {
            f1 = i;
            f2 = num / i;
            const real_t dif1 = static_cast<real_t>(f1) / static_cast<real_t>(f2);
            const real_t cur_d = std::abs(dif0 - dif1);

            if (cur_d < dd) {
                dd = cur_d;
                f1o = f1;
                f2o = f2;
            } else {
                return {f1o, f2o};
            }
        }
    }
    return {f1o, f2o};
}