#pragma once

#include "config.h"
#include "logger.h"
#include "parallel.h"
#include "utils.h"

#include <Eigen/Core>
#include <memory>
#include <string>

class Topography {
public:
    // Singleton access — mirrors InputParams::read() / IP() pattern.
    static void read(const std::string& filepath) {
        auto& mpi = Parallel::mpi();
        if (mpi.is_main()) {
            get_topo_ptr() = std::make_unique<Topography>(filepath);
        } else {
            get_topo_ptr() = std::make_unique<Topography>();
        }
        get_topo_ptr()->bcast();
    }

    static Topography& Topo() {
        auto* ptr = get_topo_ptr().get();
        if (!ptr) throw std::runtime_error("Topography: call read() first");
        return *ptr;
    }

    explicit Topography(const std::string& filepath);
    Topography() = default;

    // Write to HDF5 file
    void write(const std::string& filepath) const;

    // Apply Gaussian smoothing (sigma in degrees) to raw topo.
    Eigen::MatrixX<real_t> smooth(real_t sigma);

    // Interpolate topo onto model grid xgrids x ygrids (lon/lat in degrees).
    void grid(const Eigen::VectorX<real_t>& xgrids,
              const Eigen::VectorX<real_t>& ygrids);

    // Calculate the dip angle (degrees) at each grid point, based on the gridded topo (z).
    Eigen::MatrixX<real_t> calc_dip_angle();

    // rotate the topography by a specified angle (degrees) around a center point (clon, clat).
    void rotate(const real_t xmin, const real_t xmax,
                const real_t ymin, const real_t ymax,
                const real_t clat, const real_t clon, const real_t angle);

    inline void copy();
    
    Eigen::VectorX<real_t> lon_raw, lat_raw;  // raw topo grid coordinates
    Eigen::VectorX<real_t> lon, lat;  // raw topo grid coordinates
    Eigen::MatrixX<real_t> topo;              // topography (km, after unit conversion)
    Eigen::MatrixX<real_t> z;                 // topo interpolated onto model grid (km)
    real_t dx{0}, dy{0};                      // model grid spacing (degrees)
    

private:
    static std::unique_ptr<Topography>& get_topo_ptr() {
        static std::unique_ptr<Topography> inst;
        return inst;
    }
    void bcast();
    void read_from_file();

    void check_bounds(const Eigen::VectorX<real_t>& xgrids,
                      const Eigen::VectorX<real_t>& ygrids);

    std::string topo_file_;
};