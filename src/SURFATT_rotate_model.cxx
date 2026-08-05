#include "h5io.h"
#include "argparser.h"
#include "rapidcsv.h"
#include "sph2loc.h"
#include "logger.h"
#include "parallel.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ExportField {
    std::string name;
    std::vector<real_t> values;
};

std::string companion_dataset_key(const std::string &key,
                                  const std::string &companion_name) {
    const std::string base = rotate_model_base_field(key);
    const auto pos = key.find(base);
    if (pos == std::string::npos) {
        throw std::runtime_error("Cannot determine the base field in dataset '" + key + "'");
    }
    std::string companion = key;
    companion.replace(pos, base.size(), companion_name);
    return companion;
}

} // namespace


int rotate_model(const RotateModelArgs &args, ATTLogger &logger) {
    logger.Info("Rotating model and exporting CSV...", MODULE_MAIN);

    // read model from hdf5 file
    if (args.key.has_value()) {
        const std::string base = rotate_model_base_field(args.key.value());
        const auto matches = [&base](const char *name) { return base == name; };
        if (!std::any_of(rotate_model_fields.begin(), rotate_model_fields.end(), matches)) {
            logger.Error(fmt::format(
                "Unsupported dataset key '{}'. Allowed model fields: {}",
                args.key.value(), rotate_model_fields_str()), MODULE_MAIN);
            return EXIT_FAILURE;
        }
    }

    H5IO file(args.fname, H5IO::RDONLY);
    for (const char *axis : {"x", "y", "z"}) {
        if (!file.exists(axis)) {
            logger.Error(fmt::format(
                "Required coordinate dataset key '{}' was not found in HDF5 file '{}'",
                axis, args.fname), MODULE_MAIN);
            return EXIT_FAILURE;
        }
    }
    auto x_vec = file.read_vector<real_t>("x");
    auto y_vec = file.read_vector<real_t>("y");
    auto z_vec = file.read_vector<real_t>("z");
    int ngrid_i = static_cast<int>(x_vec.size());
    int ngrid_j = static_cast<int>(y_vec.size());
    int ngrid_k = static_cast<int>(z_vec.size());
    int nelem = ngrid_i * ngrid_j * ngrid_k;
    logger.Info(fmt::format("Loaded model grid: {} x {} x {}", ngrid_i, ngrid_j, ngrid_k), MODULE_MAIN);

    // Read every model field written by ModelGrid that is actually present.
    // With -k, retain the existing single-dataset behaviour.
    auto read_field = [&](const std::string &dataset,
                          const std::string &column_name) {
        hsize_t ni = 0, nj = 0, nk = 0;
        auto values = file.read_volume<real_t>(dataset, ni, nj, nk);
        if (ni != static_cast<hsize_t>(ngrid_i) ||
            nj != static_cast<hsize_t>(ngrid_j) ||
            nk != static_cast<hsize_t>(ngrid_k)) {
            throw std::runtime_error(fmt::format(
                "Dataset '{}' shape ({}, {}, {}) does not match model grid ({}, {}, {})",
                dataset, ni, nj, nk, ngrid_i, ngrid_j, ngrid_k));
        }
        return ExportField{column_name, std::move(values)};
    };

    std::vector<ExportField> fields;
    if (args.key.has_value()) {
        const std::string &key = args.key.value();
        if (!file.exists(key)) {
            logger.Error(fmt::format(
                "Dataset key '{}' was not found in HDF5 file '{}'",
                key, args.fname), MODULE_MAIN);
            return EXIT_FAILURE;
        }
        fields.push_back(read_field(key, rotate_model_base_field(key)));
    } else {
        for (const char *name : rotate_model_fields) {
            if (file.exists(name)) {
                fields.push_back(read_field(name, name));
            }
        }
        if (fields.empty()) {
            logger.Error(fmt::format(
                "No recognised model fields found in {}. For model_iter.h5, select "
                "an iteration dataset with -k.", args.fname), MODULE_MAIN);
            return EXIT_FAILURE;
        }
    }

    // flatten axises for rotation

    std::vector<real_t> x_flat(nelem);
    std::vector<real_t> y_flat(nelem);
    std::vector<real_t> z_flat(nelem);
    for (int i = 0; i < ngrid_i; ++i) {
        for (int j = 0; j < ngrid_j; ++j) {
            for (int k = 0; k < ngrid_k; ++k) {
                x_flat[I2V(i, j, k)] = x_vec[i];
                y_flat[I2V(i, j, k)] = y_vec[j];
                z_flat[I2V(i, j, k)] = z_vec[k];
            }
        }
    }

    // assign x/y grids to Eigen vectors for rotation
    Eigen::VectorX<real_t> x_ev = Eigen::Map<Eigen::VectorX<real_t>>(x_flat.data(), nelem);
    Eigen::VectorX<real_t> y_ev = Eigen::Map<Eigen::VectorX<real_t>>(y_flat.data(), nelem);
    Eigen::VectorX<real_t> new_x_ev = Eigen::VectorX<real_t>(nelem);
    Eigen::VectorX<real_t> new_y_ev = Eigen::VectorX<real_t>(nelem);

    if (!args.center.has_value()) {
        logger.Error("Rotation center is required: please provide -c clat/clon", MODULE_MAIN);
        return EXIT_FAILURE;
    }

    sph2loc::rtp_rotation_reverse(
        y_ev, x_ev,
        args.center.value()[0], args.center.value()[1], -args.angle,
        new_y_ev, new_x_ev
    );

    auto find_field = [&](const std::string &name) -> ExportField* {
        const auto it = std::find_if(fields.begin(), fields.end(), [&](const ExportField &field) {
            return field.name == name;
        });
        return it == fields.end() ? nullptr : &*it;
    };

    // theta is the axis of a 2-theta anisotropy and therefore has a 180-degree
    // period.  gc/gs are its cosine/sine components and must be rotated by twice
    // the frame angle.  All other model fields are scalars under this rotation.
    const real_t angle = static_cast<real_t>(args.angle);
    if (auto *theta = find_field("theta")) {
        const real_t period = static_cast<real_t>(180.0);
        for (real_t &value : theta->values) {
            value = std::fmod(std::fmod(value + angle, period) + period, period);
        }
    }

    auto rotate_gc_gs = [&](std::vector<real_t> &gc, std::vector<real_t> &gs) {
        const real_t cos_2a = std::cos(_2_CR * angle * DEG2RAD);
        const real_t sin_2a = std::sin(_2_CR * angle * DEG2RAD);
        for (int i = 0; i < nelem; ++i) {
            const real_t gc_old = gc[i];
            const real_t gs_old = gs[i];
            gc[i] = gc_old * cos_2a - gs_old * sin_2a;
            gs[i] = gc_old * sin_2a + gs_old * cos_2a;
        }
    };

    ExportField *gc = find_field("gc");
    ExportField *gs = find_field("gs");
    if (gc != nullptr && gs != nullptr) {
        rotate_gc_gs(gc->values, gs->values);
    } else if (gc != nullptr || gs != nullptr) {
        if (!args.key.has_value()) {
            logger.Error("Datasets 'gc' and 'gs' must both be present to rotate "
                         "azimuthal anisotropy", MODULE_MAIN);
            return EXIT_FAILURE;
        }
        // In single-field mode, read the paired component only for the tensor
        // transformation; it is not added to the CSV unless -k was omitted.
        const bool exporting_gc = gc != nullptr;
        const std::string &key = args.key.value();
        const std::string companion_name = exporting_gc ? "gs" : "gc";
        const std::string companion_key = companion_dataset_key(key, companion_name);
        if (!file.exists(companion_key)) {
            logger.Error(fmt::format(
                "Dataset '{}' requires companion dataset '{}' for rotation",
                key, companion_key), MODULE_MAIN);
            return EXIT_FAILURE;
        }
        ExportField companion = read_field(companion_key, companion_name);
        if (exporting_gc) {
            rotate_gc_gs(gc->values, companion.values);
        } else {
            rotate_gc_gs(companion.values, gs->values);
        }
    }
    logger.Info("Rotation completed.", MODULE_MAIN);

    std::vector<real_t> new_x(new_x_ev.data(), new_x_ev.data() + nelem);
    std::vector<real_t> new_y(new_y_ev.data(), new_y_ev.data() + nelem);

    // write to csv file
    rapidcsv::Document doc;
    doc.SetColumn<std::string>(0, fmt_col(new_x.data(), nelem, 4));
    doc.SetColumn<std::string>(1, fmt_col(new_y.data(), nelem, 4));
    doc.SetColumn<std::string>(2, fmt_col(z_flat.data(), nelem, 4));
    doc.SetColumnName(0, "lon");
    doc.SetColumnName(1, "lat");
    doc.SetColumnName(2, "depth");

    int icol = 3;
    for (const auto &field : fields) {
        doc.SetColumn<std::string>(icol, fmt_col(field.values.data(), nelem, 6));
        doc.SetColumnName(icol, field.name);
        ++icol;
    }

    doc.Save(args.outfname);
    logger.Info(fmt::format("CSV written to {} with {} model field(s)",
                            args.outfname, fields.size()), MODULE_MAIN);
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    auto args = argparse_rotate_model(argc, argv);

    // initialise MPI
    Parallel::init();
    auto &mpi = Parallel::mpi();

    // logger
    ATTLogger::init("", /*log_level=*/2, /*console_only=*/true);
    auto &logger = ATTLogger::logger();

    if (mpi.size() > 1) {
        logger.Error("SURFATT_rotate_model is not designed for parallel execution. Please run with a single process.", MODULE_MAIN);
        mpi.finalize();
        return EXIT_FAILURE;
    }

    // Prevent the HDF5 library from printing its own diagnostic stack.  Errors
    // are caught below and reported once through the application logger.
    H5::Exception::dontPrint();
    try {
        return rotate_model(args, logger);
    } catch (const H5::FileIException &e) {
        logger.Error(fmt::format(
            "Failed to open HDF5 model file '{}': {}",
            args.fname, e.getDetailMsg()), MODULE_MAIN);
    } catch (const H5::Exception &e) {
        logger.Error(fmt::format(
            "HDF5 error while processing model file '{}': {}",
            args.fname, e.getDetailMsg()), MODULE_MAIN);
    } catch (const std::exception &e) {
        logger.Error(fmt::format(
            "Failed to rotate model file '{}': {}",
            args.fname, e.what()), MODULE_MAIN);
    } catch (...) {
        logger.Error(fmt::format(
            "Unknown error while processing model file '{}'", args.fname), MODULE_MAIN);
    }

    mpi.finalize();
    return EXIT_FAILURE;
}
