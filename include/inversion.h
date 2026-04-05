#pragma once

#include "config.h"
#include "parallel.h"
#include "input_params.h"
#include "logger.h"
#include "model_grid.h"
#include "surf_grid.h"
#include "decomposer.h"
#include "preproc.h"
#include "postproc.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <stdexcept>


class Inversion {
public:
    static void init() {
        get_instance_ptr() = std::make_unique<Inversion>();
    }
    static Inversion& INV() {
        auto* ptr = get_instance_ptr().get();
        if (!ptr) throw std::runtime_error("Inversion: call init() first");
        return *ptr;
    }

    Inversion(const Inversion&)            = delete;
    Inversion& operator=(const Inversion&) = delete;

    Inversion();
    ~Inversion() = default;

    // Run only the forward calculation (no kernel accumulation or model update).
    void run_forward();

    // Run one full inversion iteration: forward + adjoint + kernel combination + model update.
    void run_inversion();

private:
    static std::unique_ptr<Inversion>& get_instance_ptr() {
        static std::unique_ptr<Inversion> instance;
        return instance;
    }

    real_t run_forward_adjoint(const bool is_calc_adj, const bool in_line_search = false);
    void init_iteration();
    void steepest_descent();
    void grad_normalization(FieldVec &grads);
    void store_model();
    void store_gradient();
    bool check_convergence();
    void model_update(FieldVec &dir);
    bool line_search();

    Tensor3r model_update_;
    FieldVec ker_curr_, ker_prev_;
    FieldVec gradient_;
    Tensor3r search_direction_;

    std::vector<real_t> misfit_ = std::vector<real_t>(InputParams::IP().inversion().niter, _0_CR);
    real_t misfit_trial_ = _0_CR;
    int    iter_ = 0;
    int    iter_start_ = 0;
    real_t alpha_, alpha_R_, alpha_L_;
};