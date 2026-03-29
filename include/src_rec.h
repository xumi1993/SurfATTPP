#pragma once

#include "config.h"
#include "rapidcsv.h"
#include "input_params.h"
#include "logger.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <mpi.h>
#include <map>
#include <string>
#include <vector>

struct event_info {
    real_t evla;
    real_t evlo;
    std::vector<int> rec_indices;
};

struct Stations {
    std::vector<std::string> stnm;
    std::vector<real_t> stla;
    std::vector<real_t> stlo;
};

struct PeriodInfo {
    Eigen::VectorX<real_t> periods;
    Eigen::VectorX<real_t> meanvel;
    std::vector<int> n_obs;
    int nperiod;
};

class SrcRec {
public:
    // Phase-velocity observation table
    static SrcRec& SR_ph() {
        static SrcRec inst;
        return inst;
    }

    // Group-velocity observation table
    static SrcRec& SR_gr() {
        static SrcRec inst;
        return inst;
    }

    static Stations& stas() {
        static Stations inst;
        return inst;
    }

    // Build the shared station list (intersection of SR_ph and SR_gr).
    // Call after loading both tables. Broadcasts result to all ranks.
    static void build_stas();

    SrcRec() = default;
    SrcRec(const SrcRec&)            = delete;
    SrcRec& operator=(const SrcRec&) = delete;
    ~SrcRec() { release_shm(); }

    // Load source-receiver table from CSV.
    // Rank 0 reads the file; each field gets its own per-node MPI shared-memory
    // window via Parallel::alloc_shared (1-D for numeric, 2-D for strings).
    // Must call release_shm() before MPI_Finalize().
    void load(const std::string& filepath);

    // Free all shared-memory windows. Call before MPI_Finalize().
    void release_shm();

    // write the source-receiver table to a CSV file (for debugging)
    void write(const std::string& filepath, const bool is_fwd = false);

    // Number of rows after load()
    int n_obs = 0;

    // Maximum length (including null terminator) for evtname / staname entries
    static constexpr int MAX_STR_LEN = 256;

    // Numeric fields — 1-D shared pointers; access as stla[i], stlo[i], ...
    real_t* stla      = nullptr;
    real_t* stlo      = nullptr;
    real_t* evla      = nullptr;
    real_t* evlo      = nullptr;
    real_t* dist      = nullptr;
    real_t* period_all = nullptr;
    real_t* tt        = nullptr;
    real_t* vel       = nullptr;
    real_t* weight    = nullptr;
    real_t* tt_fwd    = nullptr;  // forward-modeled travel time (computed in surfdisp.cpp)
    Eigen::VectorX<real_t> vel_fwd;  // for convenient access in Eigen

    std::vector<std::string> evtname;  
    std::vector<std::string> staname;
    std::map<std::string, event_info> events;  // map from event name to list of row indices for that event
    std::map<std::string, event_info> events_local; 
    std::vector<std::string> src_name_list_local;
    std::vector<std::string> src_name_list;
    PeriodInfo periods_info;

private:
    // One MPI_Win per numeric field
    MPI_Win win_stla_      = MPI_WIN_NULL;
    MPI_Win win_stlo_      = MPI_WIN_NULL;
    MPI_Win win_evla_      = MPI_WIN_NULL;
    MPI_Win win_evlo_      = MPI_WIN_NULL;
    MPI_Win win_dist_      = MPI_WIN_NULL;
    MPI_Win win_period_    = MPI_WIN_NULL;
    MPI_Win win_tt_        = MPI_WIN_NULL;
    MPI_Win win_vel_       = MPI_WIN_NULL;
    MPI_Win win_weight_    = MPI_WIN_NULL;

    // header name
    std::vector<std::string> header_names = {
        "tt", "staname", "stla", "stlo",
        "evtname", "evla", "evlo", "period",
        "weight", "dist", "vel"
    };

    void get_events();
    void get_periods();
    int nsrc_total = 0;

};
