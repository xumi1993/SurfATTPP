#pragma once
#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "src_rec.h"
#include "model_grid.h"
#include "surf_grid.h"


namespace preproc {

// 高效 period 切片辅助函数：
// 输入buf为[i][j][iper]展平，返回指定iper的[ni,nj]矩阵（RowMajor，连续内存）
inline Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
extract_period_ij(const real_t* buf, int np, int iper) {
    using MatRM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Strd  = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    using MapMatConst = Eigen::Map<const MatRM, 0, Strd>;
    // 步长: 行间隔=nj*np, 列间隔=np
    const Strd stride(ngrid_j * np, np);
    // 零拷贝视图（不连续）
    MapMatConst view(buf + iper, ngrid_i, ngrid_j, stride);
    // 拷贝成连续RowMajor矩阵，便于后续数值核
    return view;
}


inline void forward_for_event(SrcRec& sr, SurfGrid& sg) {
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();
    auto& mpi = Parallel::mpi();

    for (auto evt : sr.events_local) {
        real_t evla = evt.second.evla;
        real_t evlo = evt.second.evlo;
        int iper = evt.second.iper;

        auto svel_ij = extract_period_ij(sg.svel, sg.nperiod, iper);
        auto m11_ij = extract_period_ij(sg.m11, sg.nperiod, iper);
        auto m12_ij = extract_period_ij(sg.m12, sg.nperiod, iper);
        auto m22_ij = extract_period_ij(sg.m22, sg.nperiod, iper);
        // Compute the travel time for this source-receiver pair and period
        // using the surface wave dispersion grid (sg) and model grid (mg).
        // This is a placeholder; the actual computation would involve
        // ray tracing or finite-difference methods based on sg and mg.
        Eigen::MatrixX<real_t> tfield = FSM_UW_PS_lonlat_2d(
            mg.xgrids, mg.ygrids,
            m11_ij, m22_ij, -m12_ij,
            _1_CR / svel_ij, evlo, evla
        );
        // interpolate tfield at receiver locations and store in evt.syn_data
        for (int i = 0; i < evt.second.rec_indices.size(); ++i) {
            int rec_idx = evt.second.rec_indices[i];
            real_t stla = sr.stla[rec_idx];
            real_t stlo = sr.stlo[rec_idx];
            // Bilinear interpolation of tfield at (stlo, stla)
            evt.second.syn_data[i] = bilinear_interpolation(
                mg.xgrids.data(), mg.ygrids.data(),
                ngrid_i, ngrid_j, tfield, stlo, stla
            );
        }
    }
    mpi.barrier();
}

inline void run_forward_adjoint(const bool is_calc_adj){
    auto& IP = InputParams::IP();
    auto& mg = ModelGrid::MG();

   for (surfType itype : {surfType::PH, surfType::GR}) {
        if (!IP.data().vel_type[itype]) continue;
        auto &sg = (itype == 0) ? SurfGrid::SG_ph() : SurfGrid::SG_gr();
        auto &sr = (itype == 0) ? SrcRec::SR_ph() : SrcRec::SR_gr();

        // Compute surface wave dispersion from the 3D S-wave velocity model.
        sg.fwdsurf();

        // calculate travel time for each source-receiver pair and period, and store in sr.events_local
        forward_for_event(sr, sg);
    }
}
}