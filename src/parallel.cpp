#include "parallel.h"

#include <cstring>
#include <numeric>

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

void Parallel::init() {
    get_instance_ptr() = std::unique_ptr<Parallel>(new Parallel());
}

Parallel &Parallel::mpi() {
    auto *ptr = get_instance_ptr().get();
    if (!ptr) throw std::runtime_error("Parallel: call init() first");
    return *ptr;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Parallel::Parallel() {
    int ier = MPI_Init(NULL, NULL);
    if (ier != MPI_SUCCESS)
        throw std::runtime_error("Parallel: MPI_Init failed");

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    comm_ = MPI_COMM_WORLD;

    // Create a shared-memory communicator to identify processes on the same node
    MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_,
                        MPI_INFO_NULL, &node_comm_);
    MPI_Comm_rank(node_comm_, &local_rank_);
    MPI_Comm_size(node_comm_, &local_size_);

    build_rank_map();
}

// ---------------------------------------------------------------------------
// Build the global ↔ local rank map (mirrors init_mpi rank_map construction)
// Each entry: {global_rank, local_rank_on_node}
// ---------------------------------------------------------------------------

void Parallel::build_rank_map() {
    rank_map_.resize(size_, {0, 0});

    // Each process contributes its own entry
    std::vector<int> local_data(size_ * 2, 0);
    local_data[rank_ * 2]     = rank_;
    local_data[rank_ * 2 + 1] = local_rank_;

    std::vector<int> global_data(size_ * 2, 0);
    MPI_Allreduce(local_data.data(), global_data.data(),
                  size_ * 2, MPI_INT, MPI_SUM, comm_);

    for (int i = 0; i < size_; ++i) {
        rank_map_[i][0] = global_data[i * 2];
        rank_map_[i][1] = global_data[i * 2 + 1];
    }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void Parallel::finalize() {
    MPI_Barrier(MPI_COMM_WORLD);
    int ier = MPI_Finalize();
    if (ier != MPI_SUCCESS)
        throw std::runtime_error("Parallel: MPI_Finalize failed");
}

void Parallel::abort() {
    MPI_Abort(MPI_COMM_WORLD, 30);
    std::exit(30);  // unreachable, but satisfies [[noreturn]]
}

// ---------------------------------------------------------------------------
// Synchronisation
// ---------------------------------------------------------------------------

void Parallel::barrier() {
    int ier = MPI_Barrier(comm_);
    if (ier != MPI_SUCCESS)
        throw std::runtime_error("Parallel: MPI_Barrier failed");
}

void Parallel::wait(MPI_Request &req) {
    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

// ---------------------------------------------------------------------------
// Broadcast — bool and string specialisations
// ---------------------------------------------------------------------------

void Parallel::bcast(bool &val) {
    int tmp = val ? 1 : 0;
    MPI_Bcast(&tmp, 1, MPI_INT, 0, comm_);
    val = (tmp != 0);
}

void Parallel::bcast(bool *buf, int count) {
    std::vector<int> tmp(buf, buf + count);
    MPI_Bcast(tmp.data(), count, MPI_INT, 0, comm_);
    for (int i = 0; i < count; ++i) buf[i] = (tmp[i] != 0);
}

void Parallel::bcast(std::string &s) {
    int len = static_cast<int>(s.size());
    MPI_Bcast(&len, 1, MPI_INT, 0, comm_);
    s.resize(len);
    MPI_Bcast(s.data(), len, MPI_CHAR, 0, comm_);
}

// ---------------------------------------------------------------------------
// Allreduce helpers
// ---------------------------------------------------------------------------

void Parallel::any_all(bool send, bool &recv) {
    int s = send ? 1 : 0, r = 0;
    MPI_Allreduce(&s, &r, 1, MPI_INT, MPI_LOR, comm_);
    recv = (r != 0);
}

// ---------------------------------------------------------------------------
// Work distribution
// ---------------------------------------------------------------------------

void Parallel::scatter_range(int total, int &istart, int &iend) const {
    if (total < size_) {
        if (rank_ < total) { istart = rank_ + 1; iend = istart; }
        else               { istart = 1; iend = 0; }
        return;
    }
    int count     = total / size_;
    int remainder = total % size_;
    if (rank_ < remainder) {
        istart = rank_ * (count + 1) + 1;
        iend   = istart + count;
    } else {
        istart = rank_ * count + remainder + 1;
        iend   = istart + count - 1;
    }
}
