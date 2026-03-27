#pragma once

#include "config.h"
#include <mpi.h>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Compile-time MPI type trait  (maps C++ type → MPI_Datatype)
// ---------------------------------------------------------------------------
template<typename T> inline MPI_Datatype mpi_type_of();
template<> inline MPI_Datatype mpi_type_of<int>()    { return MPI_INT; }
template<> inline MPI_Datatype mpi_type_of<float>()  { return MPI_FLOAT; }
template<> inline MPI_Datatype mpi_type_of<double>() { return MPI_DOUBLE; }

// Paired MPI type for MPI_MAXLOC / MPI_MINLOC operations
template<typename T> inline MPI_Datatype mpi_maxloc_type_of();
template<> inline MPI_Datatype mpi_maxloc_type_of<float>()  { return MPI_FLOAT_INT; }
template<> inline MPI_Datatype mpi_maxloc_type_of<double>() { return MPI_DOUBLE_INT; }

// ---------------------------------------------------------------------------
// Parallel — thin singleton wrapper around MPI, mirroring parallel.f90
//
// Usage:
//   Parallel::init();          // call once in main()
//   auto &mpi = Parallel::mpi();
//   if (mpi.is_main()) { ... }
// ---------------------------------------------------------------------------
class Parallel {
public:
    // ---- Singleton ---------------------------------------------------------
    static void init();
    static Parallel &mpi();
    bool is_main()       const { return rank_ == 0; }
    bool is_node_main()  const { return local_rank_ == 0; }
    inline int select_rank_for_src(const int& id_src){
        return id_src % size_;
    }

    // ---- Basic query -------------------------------------------------------
    int      rank()       const { return rank_; }
    int      size()       const { return size_; }
    int      local_rank() const { return local_rank_; }
    int      local_size() const { return local_size_; }
    MPI_Comm comm()       const { return comm_; }
    MPI_Comm node_comm()  const { return node_comm_; }

    // ---- Lifecycle ---------------------------------------------------------
    ~Parallel() { finalize(); }
    void finalize();
    [[noreturn]] void abort();

    // ---- Synchronisation ---------------------------------------------------
    void barrier();
    double wtime() const { return MPI_Wtime(); }
    void wait(MPI_Request &req);

    // ---- Broadcast (from rank 0) -------------------------------------------
    // scalar
    template<typename T>
    void bcast(T &val) {
        MPI_Bcast(&val, 1, mpi_type_of<T>(), 0, comm_);
    }
    // array
    template<typename T>
    void bcast(T *buf, int count) {
        MPI_Bcast(buf, count, mpi_type_of<T>(), 0, comm_);
    }
    // bool scalar
    void bcast(bool &val);
    // bool array
    void bcast(bool *buf, int count);
    // string
    void bcast(std::string &s);

    template<typename T>
    inline void bcast_vec(std::vector<T> &v) {
        int sz = static_cast<int>(v.size());
        bcast(sz);
        v.resize(sz);
        bcast(v.data(), sz);
    }

    // std::vector<bool> is special — proxy elements, not real bools.
    inline void bcast_bool_vec(std::vector<bool> &v) {
        int sz = static_cast<int>(v.size());
        bcast(sz);
        v.resize(sz);
        std::vector<int> tmp(sz);
        if (is_main())
            for (int i = 0; i < sz; ++i) tmp[i] = v[i] ? 1 : 0;
        bcast(tmp.data(), sz);
        if (!is_main())
            for (int i = 0; i < sz; ++i) v[i] = (tmp[i] != 0);
    }


    // ---- Reduce to rank 0  (min / max / sum) --------------------------------
    template<typename T>
    void min_all(T send, T &recv) {
        MPI_Reduce(&send, &recv, 1, mpi_type_of<T>(), MPI_MIN, 0, comm_);
    }
    template<typename T>
    void max_all(T send, T &recv) {
        MPI_Reduce(&send, &recv, 1, mpi_type_of<T>(), MPI_MAX, 0, comm_);
    }
    template<typename T>
    void sum_all(T send, T &recv) {
        MPI_Reduce(&send, &recv, 1, mpi_type_of<T>(), MPI_SUM, 0, comm_);
    }
    // nD arrays: reduce to rank 0
    template<typename T>
    void sum_all(const T *send, T *recv, int count) {
        MPI_Reduce(send, recv, count, mpi_type_of<T>(), MPI_SUM, 0, comm_);
    }

    template<typename T>
    void sum_all_vect_inplace(std::vector<T> &buf, int count) {
        MPI_Reduce(MPI_IN_PLACE, buf.data(), buf.size(), mpi_type_of<T>(), MPI_SUM, 0, comm_);
    }

    // ---- Allreduce  (min / max / sum) ---------------------------------------
    template<typename T>
    void min_all_all(T send, T &recv) {
        MPI_Allreduce(&send, &recv, 1, mpi_type_of<T>(), MPI_MIN, comm_);
    }
    template<typename T>
    void max_all_all(T send, T &recv) {
        MPI_Allreduce(&send, &recv, 1, mpi_type_of<T>(), MPI_MAX, comm_);
    }
    template<typename T>
    void sum_all_all(T send, T &recv) {
        MPI_Allreduce(&send, &recv, 1, mpi_type_of<T>(), MPI_SUM, comm_);
    }
    // nD arrays: allreduce
    template<typename T>
    void sum_all_all(const T *send, T *recv, int count) {
        MPI_Allreduce(send, recv, count, mpi_type_of<T>(), MPI_SUM, comm_);
    }
    // in-place allreduce (max)
    template<typename T>
    void max_allreduce(T *buf, int count) {
        std::vector<T> tmp(buf, buf + count);
        MPI_Allreduce(tmp.data(), buf, count, mpi_type_of<T>(), MPI_MAX, comm_);
    }

    template<typename T>
    void sum_all_all_vect_inplace(std::vector<T> &buf) {
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), buf.size(), mpi_type_of<T>(), MPI_SUM, comm_);
    }
    // maxloc (returns {value, rank} pair)
    template<typename T>
    void maxloc_all(T send_val, int send_rank, T &recv_val, int &recv_rank) {
        struct { T val; int rank; } in{send_val, send_rank}, out{};
        MPI_Allreduce(&in, &out, 1, mpi_maxloc_type_of<T>(), MPI_MAXLOC, comm_);
        recv_val  = out.val;
        recv_rank = out.rank;
    }
    // logical OR across all ranks
    void any_all(bool send, bool &recv);

    // ---- Async send / recv -------------------------------------------------
    template<typename T>
    void isend(const T *buf, int count, int dest, int tag, MPI_Request &req) {
        MPI_Isend(buf, count, mpi_type_of<T>(), dest, tag, comm_, &req);
    }
    template<typename T>
    void irecv(T *buf, int count, int src, int tag, MPI_Request &req) {
        MPI_Irecv(buf, count, mpi_type_of<T>(), src, tag, comm_, &req);
    }

    // ---- Blocking send / recv ----------------------------------------------
    template<typename T>
    void send(const T *buf, int count, int dest) {
        MPI_Send(buf, count, mpi_type_of<T>(), dest, MPI_TAG_BASE, comm_);
    }
    template<typename T>
    void recv(T *buf, int count, int src) {
        MPI_Recv(buf, count, mpi_type_of<T>(), src, MPI_TAG_BASE, comm_,
                 MPI_STATUS_IGNORE);
    }

    void send(std::string &s, int dest) {
        int len = static_cast<int>(s.size());
        MPI_Send(&len, 1, MPI_INT, dest, MPI_TAG_BASE, comm_);
        MPI_Send(s.data(), len, MPI_CHAR, dest, MPI_TAG_BASE, comm_);
    }

    void recv(std::string &s, int src) {
        int len;
        MPI_Recv(&len, 1, MPI_INT, src, MPI_TAG_BASE, comm_, MPI_STATUS_IGNORE);
        s.resize(len);
        MPI_Recv(s.data(), len, MPI_CHAR, src, MPI_TAG_BASE, comm_, MPI_STATUS_IGNORE);
    }

    // ---- Gather  (to rank 0) ------------------------------------------------
    template<typename T>
    void gather_all(const T *send, int sendcnt,
                    T *recv,       int recvcnt) {
        MPI_Gather(send, sendcnt, mpi_type_of<T>(),
                   recv, recvcnt, mpi_type_of<T>(), 0, comm_);
    }
    // scalar gather
    template<typename T>
    void gather_all(T send, T *recv) {
        MPI_Gather(&send, 1, mpi_type_of<T>(),
                    recv, 1, mpi_type_of<T>(), 0, comm_);
    }
    // Allgather
    template<typename T>
    void gather_all_all(const T *send, int sendcnt,
                        T *recv,       int recvcnt) {
        MPI_Allgather(send, sendcnt, mpi_type_of<T>(),
                      recv, recvcnt, mpi_type_of<T>(), comm_);
    }

    // ---- Variable-length gather (gatherv, to rank 0) -----------------------
    template<typename T>
    void gatherv_all(const T *send,  int sendcnt,
                     T *recv,
                     const int *recvcounts,
                     const int *displs,
                     int        total) {
        MPI_Gatherv(send, sendcnt,         mpi_type_of<T>(),
                    recv, recvcounts, displs, mpi_type_of<T>(),
                    0, comm_);
    }

    // ---- Work distribution (scatter_all_i equivalent) ----------------------
    // Divides [1..total] across all ranks; returns [istart, iend] (1-based,
    // inclusive), matching the Fortran scatter_all_i convention.
    void scatter_range(int total, int &istart, int &iend) const;

    // ---- Sync from main rank (node-aware, mirrors sync_from_main_rank_*) ---
    // Sends `buf` from global rank 0 to the lead process (local_rank==0) of
    // every other node, then barriers.  All other intra-node distribution is
    // left to the caller (or use bcast within the node communicator).
    template<typename T>
    void sync_from_main_rank(T *buf, int count) {
        if (rank_ == 0) {
            for (int i = 1; i < size_; ++i) {
                if (rank_map_[i] == 0)          // head of another node
                    send(buf, count, i);
            }
        } else if (local_rank_ == 0) {
            recv(buf, count, 0);
        }
        barrier();
    }

    // ---- Shared-memory window allocation -----------------------------------
    // Mirrors prepare_shm_array_cr_1d / prepare_shm_array_cr_2d.
    // node_rank 0 allocates the full array; others allocate 0 bytes and
    // retrieve the node-rank-0 base pointer via MPI_Win_shared_query.
    // MPI_Win_fence(0) is called at the end to open the first RMA epoch.
    //
    // 1-D overload — alloc_shared(n_elem, buf, win)
    template<typename T>
    void alloc_shared(int n_elem, T*& buf, MPI_Win& win) {
        int count = (local_rank_ == 0) ? n_elem : 0;
        T* base_ptr = nullptr;
        MPI_Win_allocate_shared(
            static_cast<MPI_Aint>(count) * static_cast<MPI_Aint>(sizeof(T)),
            static_cast<int>(sizeof(T)), MPI_INFO_NULL, node_comm_,
            &base_ptr, &win);
        int *model = nullptr, flag = 0;
        MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
        if (local_rank_ != 0) {
            MPI_Aint sz_dummy = 0;  int disp_dummy = 0;
            MPI_Win_shared_query(win, 0, &sz_dummy, &disp_dummy, &base_ptr);
        }
        buf = base_ptr;
        MPI_Win_fence(0, win);
    }

    // 2-D overload — alloc_shared(nx, ny, buf, win)
    // buf[i][j] == base_ptr[i * ny + j].  Row-pointer array allocated with new[].
    template<typename T>
    void alloc_shared(int nx, int ny, T**& buf, MPI_Win& win) {
        int count = (local_rank_ == 0) ? nx * ny : 0;
        T* base_ptr = nullptr;
        MPI_Win_allocate_shared(
            static_cast<MPI_Aint>(count) * static_cast<MPI_Aint>(sizeof(T)),
            static_cast<int>(sizeof(T)), MPI_INFO_NULL, node_comm_,
            &base_ptr, &win);
        int *model = nullptr, flag = 0;
        MPI_Win_get_attr(win, MPI_WIN_MODEL, &model, &flag);
        if (local_rank_ != 0) {
            MPI_Aint sz_dummy = 0;  int disp_dummy = 0;
            MPI_Win_shared_query(win, 0, &sz_dummy, &disp_dummy, &base_ptr);
        }
        buf = new T*[static_cast<std::size_t>(nx)];
        for (int i = 0; i < nx; ++i)
            buf[i] = base_ptr + i * ny;
        MPI_Win_fence(0, win);
    }

    // Free a 1-D shared window.  Resets win to MPI_WIN_NULL and buf to nullptr.
    template<typename T>
    void free_shared(T*& ptr, MPI_Win& win) {
        if (win != MPI_WIN_NULL) { MPI_Win_free(&win); win = MPI_WIN_NULL; }
        ptr = nullptr;
    }

    // Free a 2-D shared window.  Also deletes the row-pointer array.
    template<typename T>
    void free_shared(T**& buf, MPI_Win& win) {
        if (win != MPI_WIN_NULL) { MPI_Win_free(&win); win = MPI_WIN_NULL; }
        delete[] buf;
        buf = nullptr;
    }

private:
    explicit Parallel();

    static std::unique_ptr<Parallel> &get_instance_ptr() {
        static std::unique_ptr<Parallel> inst;
        return inst;
    }

    int      rank_{0}, size_{1};
    int      local_rank_{0}, local_size_{1};
    MPI_Comm comm_{MPI_COMM_WORLD};
    MPI_Comm node_comm_{MPI_COMM_NULL};

    // rank_map_[i] = {global_rank, local_rank_within_node}
    std::vector<int> rank_map_;

};
