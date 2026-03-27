#include "src_rec.h"
#include "parallel.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>


// ---------------------------------------------------------------------------
void SrcRec::load(const std::string& filepath)
{
    auto& mpi = Parallel::mpi();

    std::vector<real_t> v_stla, v_stlo, v_evla, v_evlo, v_dist,
                        v_period, v_tt, v_vel, v_weight;
    std::vector<std::string> v_evtname, v_staname;

    // 1. Rank 0 reads the CSV
    if (mpi.is_main()) {
        rapidcsv::Document doc(filepath, rapidcsv::LabelParams(0, -1));
        n_obs = static_cast<int>(doc.GetRowCount());

        v_stla    = doc.GetColumn<real_t>("stla");
        v_stlo    = doc.GetColumn<real_t>("stlo");
        v_evla    = doc.GetColumn<real_t>("evla");
        v_evlo    = doc.GetColumn<real_t>("evlo");
        v_period  = doc.GetColumn<real_t>("period");
        v_tt      = doc.GetColumn<real_t>("tt");
        evtname = doc.GetColumn<std::string>("evtname");
        staname = doc.GetColumn<std::string>("staname");

        // Optional "weight" column; if missing, fill with 1.0
        try{
            v_weight  = doc.GetColumn<real_t>("weight");
        } catch (const std::exception& e) {
            // If "weight" column is missing, fill with 1.0
            v_weight.resize(n_obs, _1_CR);
        }

        // Optional "dist" column; if missing, compute from stla/stlo/evla/evlo
        try {
            v_dist    = doc.GetColumn<real_t>("dist");
        } catch (const std::exception& e) {
            v_dist.resize(n_obs);
            for (int i = 0; i < n_obs; ++i) {
                v_dist[i] = gps2dist(v_stla[i], v_stlo[i], v_evla[i], v_evlo[i]);
            }
        }

        // Optional "vel" column; if missing, compute from dist and tt
        try {
            v_vel     = doc.GetColumn<real_t>("vel");
        } catch (const std::exception& e) {
            v_vel.resize(n_obs);
            for (int i = 0; i < n_obs; ++i) {
                v_vel[i] = (v_tt[i] > 0) ? (v_dist[i] / v_tt[i]) : 0.0;
            }
        }
    }
    mpi.barrier();

    // 2. Broadcast row count
    mpi.bcast(n_obs);
    mpi.bcast(evtname);
    mpi.bcast(staname);

    // 3. One shared window per numeric field
    mpi.alloc_shared(n_obs, stla, win_stla_);
    mpi.alloc_shared(n_obs, stlo, win_stlo_);
    mpi.alloc_shared(n_obs, evla, win_evla_);
    mpi.alloc_shared(n_obs, evlo, win_evlo_);
    mpi.alloc_shared(n_obs, dist, win_dist_);
    mpi.alloc_shared(n_obs, period_all, win_period_);
    mpi.alloc_shared(n_obs, tt, win_tt_);
    mpi.alloc_shared(n_obs, vel, win_vel_);
    mpi.alloc_shared(n_obs, weight, win_weight_);
    if (mpi.is_main()) {
        std::copy(v_stla.begin(), v_stla.end(), stla);
        std::copy(v_stlo.begin(), v_stlo.end(), stlo);
        std::copy(v_evla.begin(), v_evla.end(), evla);
        std::copy(v_evlo.begin(), v_evlo.end(), evlo);
        std::copy(v_dist.begin(), v_dist.end(), dist);
        std::copy(v_period.begin(), v_period.end(), period_all);
        std::copy(v_tt.begin(), v_tt.end(), tt);
        std::copy(v_vel.begin(), v_vel.end(), vel);
        std::copy(v_weight.begin(), v_weight.end(), weight);
    }
    mpi.barrier();

    mpi.sync_from_main_rank(stla, n_obs);
    mpi.sync_from_main_rank(stlo, n_obs);
    mpi.sync_from_main_rank(evla, n_obs);
    mpi.sync_from_main_rank(evlo, n_obs);
    mpi.sync_from_main_rank(dist, n_obs);
    mpi.sync_from_main_rank(period_all, n_obs);
    mpi.sync_from_main_rank(tt, n_obs);
    mpi.sync_from_main_rank(vel, n_obs);
    mpi.sync_from_main_rank(weight, n_obs);
}

// ---------------------------------------------------------------------------
void SrcRec::release_shm()
{
    auto& par = Parallel::mpi();
    par.free_shared(stla,       win_stla_);
    par.free_shared(stlo,       win_stlo_);
    par.free_shared(evla,       win_evla_);
    par.free_shared(evlo,       win_evlo_);
    par.free_shared(dist,       win_dist_);
    par.free_shared(period_all, win_period_);
    par.free_shared(tt,         win_tt_);
    par.free_shared(vel,        win_vel_);
    par.free_shared(weight,     win_weight_);
}

