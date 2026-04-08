#include "utils.h"

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: test_interp2d_dump <output-file>\n";
        return 2;
    }

    Eigen::VectorX<real_t> xgrid(4);
    xgrid << 0.0, 0.7, 1.9, 3.5;

    Eigen::VectorX<real_t> ygrid(3);
    ygrid << -1.0, 0.2, 2.0;

    Eigen::MatrixX<real_t> z(xgrid.size(), ygrid.size());
    for (int i = 0; i < xgrid.size(); ++i) {
        for (int j = 0; j < ygrid.size(); ++j) {
            const real_t x = xgrid(i);
            const real_t y = ygrid(j);
            // Use a larger dynamic range so printed C++/SciPy values are easier to compare by eye.
            z(i, j) = 1000.0 * (std::sin(0.8 * x) + std::cos(0.6 * y) + 0.1 * x * y)
                    + 25000.0;
        }
    }

    Eigen::VectorX<real_t> xq(7);
    xq << 0.0, 0.35, 1.2, 3.5, 1.9, -0.1, 1.0;

    Eigen::VectorX<real_t> yq(7);
    yq << -1.0, -0.4, 1.1, 2.0, 0.2, 0.0, 2.1;

    Eigen::VectorX<real_t> zq = interp2d(xgrid, ygrid, z, xq, yq);

    std::ofstream out(argv[1]);
    if (!out) {
        std::cerr << "Failed to open output file: " << argv[1] << "\n";
        return 1;
    }

    out << std::setprecision(17);
    out << xgrid.size() << " " << ygrid.size() << "\n";
    for (int i = 0; i < xgrid.size(); ++i) {
        out << xgrid(i) << (i + 1 == xgrid.size() ? '\n' : ' ');
    }
    for (int j = 0; j < ygrid.size(); ++j) {
        out << ygrid(j) << (j + 1 == ygrid.size() ? '\n' : ' ');
    }
    for (int i = 0; i < z.rows(); ++i) {
        for (int j = 0; j < z.cols(); ++j) {
            out << z(i, j) << (j + 1 == z.cols() ? '\n' : ' ');
        }
    }

    out << xq.size() << "\n";
    for (int k = 0; k < xq.size(); ++k) {
        out << xq(k) << " " << yq(k) << " " << zq(k) << "\n";
    }

    return 0;
}
