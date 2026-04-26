#pragma once

#include <vector>
#include <stdexcept>

std::vector<double> disper(const float *thkm, const float *vpm, const float *vsm,
                             const float *rhom, int nlayer, int iflsph, int iwave,
                             int mode, int igr, int kmax, const double *t);