#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.h"

// ---------------------------------------------------------------------------
// ArgList — thin read-only view over the command-line token list.
//
// Provides two clean query operations instead of raw index arithmetic:
//   has(flag)     – true if the flag token appears anywhere
//   get(flag)     – std::optional<string> value that follows flag
//   require(flag) – like get(), but throws if the flag is absent
// ---------------------------------------------------------------------------
struct ArgList {
    std::vector<std::string> args;

    ArgList(int argc, char* argv[]) : args(argv + 1, argv + argc) {}

    bool empty() const { return args.empty(); }

    bool has(std::string flag) const {
        return std::any_of(args.begin(), args.end(), [&flag](const std::string& s) {
            return s == flag;
        });
    }

    std::optional<std::string> get(std::string flag) const {
        for (std::size_t i = 0; i + 1 < args.size(); ++i)
            if (args[i] == flag) return args[i + 1];
        return std::nullopt;
    }

    std::string require(std::string flag) const {
        auto v = get(flag);
        if (!v) throw std::runtime_error("required argument " + flag + " is missing");
        return *v;
    }
};

// ---------------------------------------------------------------------------
// Slash-separated value parsers
// ---------------------------------------------------------------------------

// "a/b/c" → {int, int, int}
inline std::array<int, 3> parse_3int(const std::string& s) {
    auto s1 = s.find('/');
    auto s2 = s.find('/', s1 + 1);
    if (s1 == std::string::npos || s2 == std::string::npos)
        throw std::runtime_error("expected a/b/c format, got \"" + s + "\"");
    return {
        std::stoi(s.substr(0, s1)),
        std::stoi(s.substr(s1 + 1, s2 - s1 - 1)),
        std::stoi(s.substr(s2 + 1))
    };
}

// "a/b" → {double, double}
inline std::array<double, 2> parse_2double(const std::string& s) {
    auto sl = s.find('/');
    if (sl == std::string::npos)
        throw std::runtime_error("expected a/b format, got \"" + s + "\"");
    return { std::stod(s.substr(0, sl)), std::stod(s.substr(sl + 1)) };
}

// "a/b" → {int, int}
inline std::array<int, 2> parse_2int(const std::string& s) {
    auto sl = s.find('/');
    if (sl == std::string::npos)
        throw std::runtime_error("expected a/b format, got \"" + s + "\"");
    return { std::stoi(s.substr(0, sl)), std::stoi(s.substr(sl + 1)) };
}

// "a/b/c[/d]" → {{int, int, int}, double}
// If /d is absent, angle defaults to default_angle.
inline std::pair<std::array<int, 3>, double> parse_3int_1double(
    const std::string& s, double default_angle = 120.0
) {
    auto s1 = s.find('/');
    auto s2 = s.find('/', s1 + 1);
    auto s3 = s.find('/', s2 + 1);
    if (s1 == std::string::npos || s2 == std::string::npos)
        throw std::runtime_error("expected a/b/c or a/b/c/d format, got \"" + s + "\"");

    const std::array<int, 3> nxyz = {
        std::stoi(s.substr(0, s1)),
        std::stoi(s.substr(s1 + 1, s2 - s1 - 1)),
        std::stoi(s.substr(s2 + 1, (s3 == std::string::npos ? s.size() : s3) - s2 - 1))
    };

    const double angle = (s3 == std::string::npos)
        ? default_angle
        : std::stod(s.substr(s3 + 1));

    return {
        nxyz,
        angle
    };
}

// ---------------------------------------------------------------------------
// Argument structs
// ---------------------------------------------------------------------------

struct TomoArgs {
    bool isfwd = false;
};

struct CbFwdArgs {
    std::array<int, 3> ncb      = {0, 0, 0};
    std::array<int, 3> ncb_ani  = {0, 0, 0};
    double ani_angle = 120.0;  // anisotropy fast-axis angle in degrees (for azimuthal anisotropy)
    double pert_vel  = 0.08;
    double pert_ani  = 0.0;   // azimuthal anisotropy perturbation magnitude (gc and gs)
    double pert_zeta = 0.0;   // radial anisotropy perturbation magnitude (zeta)
    double hmarg     = 0.0;
    double anom_size = 0.0;
    double max_noise = 0.0;
    bool   only_vs   = false;
    int   model_type = 0;  // 0: isotropic, 1: azimuthal anisotropy, 2: radial anisotropy
};

struct RotateSrcRecArgs {
    std::string fname;
    std::string outfname;
    real_t angle = 0.0;
    std::array<real_t, 2> center = {0.0, 0.0};
};

struct RotateTopoArgs {
    std::string fname;
    std::string outfname;
    real_t angle = 0.0;
    std::array<real_t, 2> center = {0.0, 0.0};
    std::array<real_t, 2> xrange = {0.0, 0.0};
    std::array<real_t, 2> yrange = {0.0, 0.0};
};

struct RotateModelArgs {
    std::string fname;
    std::string outfname;
    // Scalar dataset to read and export, see rotate_model_fields.  Absent when -k
    // was not supplied: the export then defaults to "vs" and appends whichever
    // anisotropy fields the file carries.  When supplied, that dataset is the only
    // model column written.
    std::optional<std::string> key;
    real_t angle = 0.0;
    std::optional<std::array<real_t, 2>> center;  // absent when not supplied on the command line
};

struct Tomo2DArgs {
    std::string fname;
    bool isfwd  = false;
    std::array<int, 2> ncb = {0, 0};
    double pert_vel = 0.08;
    double hmarg    = 0.0;
};

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------

inline void argparse_tomo(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: surfatt_tomo -i para_file [-f] [-h]\n\n"
            "Adjoint-state travel time tomography for surface wave\n\n"
            "required arguments:\n"
            "  -i para_file  Path to parameter file in yaml format\n\n"
            "optional arguments:\n"
            "  -f            Forward simulate travel time instead of inversion (default: false)\n"
            "  -h            Print help message\n";
        std::exit(0);
    }
    input_file = al.require("-i");
    if (al.has("-f")) {
        run_mode = FORWARD_ONLY;
    }
}

inline CbFwdArgs argparse_cb_fwd(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: surfatt_cb_fwd -i para_file -n nx/ny/nz [-h]"
            " [-m margin_degree] [-p pert] [-s anom_size_km]\n\n"
            "Create checkerboard and forward simulate travel time for surface wave\n\n"
            "required arguments:\n"
            "  -i para_file             Path to parameter file in yaml format\n"
            "  -n nx/ny/nz              Number of anomalies along X, Y and Z\n\n"
            "optional arguments:\n"
            "  -h                       Print help message\n"
            "  -a nx/ny/nz[/angle]      Add azimuthal anisotropy anomalies with optional angle (deg, default: 120)\n"
            "  -r                       Use radial anisotropy instead of azimuthal (cannot be used with -a)\n"
            "  -e tt_noise              Add random noise to travel time data (default: 0)\n"
            "  -v                       Only perturb Vs model, default: false\n"
            "  -m margin_degree         Margin between anomalies in degrees (default: 0)\n"
            "  -p pert_vel[/pert_ani]   Magnitude of velocity/anisotropy perturbations\n"
            "                           For azimuthal: -p pert_vel[/pert_gc_gs] (default: 0.08)\n"
            "                           For radial: -p pert_vs[/pert_zeta] (default: 0.08)\n"
            "  -s anom_size_km          Size of top anomalies in km (default: uniform)\n";
        std::exit(0);
    }
    CbFwdArgs out;
    input_file  = al.require("-i");
    out.only_vs = al.has("-v");
    if (auto v = al.get("-n")) out.ncb = parse_3int(*v);

    // Check for mutually exclusive -a and -r options
    bool has_azi_ani = al.has("-a");
    bool has_rad_ani = al.has("-r");

    if (has_azi_ani && has_rad_ani) {
        throw std::runtime_error("Options -a (azimuthal anisotropy) and -r (radial anisotropy) cannot be used together");
    }

    if (has_azi_ani) {
        auto [ncb_ani, ani_angle] = parse_3int_1double(*al.get("-a"));
        out.ncb_ani = ncb_ani;
        out.ani_angle = ani_angle;
        out.model_type = 1;
    } else if (has_rad_ani) {
        out.ncb_ani = out.ncb;  // use same as isotropic checkerboard
        out.model_type = 2;
    } else {
        out.model_type = 0;
    }

    if (auto v = al.get("-p")) {
        if (v->find('/') != std::string::npos) {
            auto pert = parse_2double(*v);
            out.pert_vel = pert[0];
            if (out.model_type == 2) {
                out.pert_zeta = pert[1];  // for radial anisotropy
            } else {
                out.pert_ani = pert[1];   // for azimuthal anisotropy
            }
        } else {
            out.pert_vel = std::stod(*v);
        }
    }
    if (auto v = al.get("-m")) out.hmarg     = std::stod(*v);
    if (auto v = al.get("-s")) out.anom_size = std::stod(*v);
    if (auto v = al.get("-e")) out.max_noise = std::stod(*v);
    run_mode = FORWARD_ONLY;
    return out;
}

inline RotateSrcRecArgs argparse_rotate_src_rec(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: surfatt_rotate_src_rec -i src_rec_file -a angle -c clat/clon [-h] [-o out_file]\n\n"
            "Rotate source and receiver locations by a given angle (anti-clockwise)\n\n"
            "required arguments:\n"
            "  -i src_rec_file      Path to src_rec file in csv format\n"
            "  -a angle             Rotation angle in degrees\n"
            "  -c clat/clon         Centre of rotation (lat/lon)\n\n"
            "optional arguments:\n"
            "  -h                   Print help message\n"
            "  -o out_file          Output file name (default: input file + \"_rot\")\n";
        std::exit(0);
    }
    RotateSrcRecArgs out;
    out.fname    = al.require("-i");
    out.outfname = out.fname + "_rot";   // default; overridden by -o if present
    out.angle    = std::stod(al.require("-a"));
    out.center   = parse_2double(al.require("-c"));
    if (auto v = al.get("-o")) out.outfname = *v;
    return out;
}

inline RotateTopoArgs argparse_rotate_topo(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: surfatt_rotate_topo -i topo_file -a angle -c clat/clon"
            " -o out_file -x xmin/xmax -y ymin/ymax [-h]\n\n"
            "Rotate topography by a given angle (anti-clockwise)\n\n"
            "required arguments:\n"
            "  -i topo_file         Path to topography file in netcdf format\n"
            "  -a angle             Rotation angle in degrees\n"
            "  -c clat/clon         Centre of rotation (lat/lon)\n"
            "  -o out_file          Output file name\n"
            "  -x xmin/xmax         New x-coordinate range\n"
            "  -y ymin/ymax         New y-coordinate range\n\n"
            "optional arguments:\n"
            "  -h                   Print help message\n";
        std::exit(0);
    }
    return {
        .fname    = al.require("-i"),
        .outfname = al.require("-o"),
        .angle    = std::stod(al.require("-a")),
        .center   = parse_2double(al.require("-c")),
        .xrange   = parse_2double(al.require("-x")),
        .yrange   = parse_2double(al.require("-y"))
    };
}

// Model fields that surfatt_rotate_model is allowed to export.  Every entry is a
// scalar field, i.e. invariant under a rotation of the horizontal frame, so only
// the grid coordinates have to be transformed and the values travel unchanged.
inline const std::array<const char*, 7> rotate_model_fields = {
    "vs", "vsv", "vsh", "vp", "rho", "zeta", "gamma"
};

// Fields that are NOT invariant under a frame rotation and must never be exported
// as a plain pass-through: gc/gs are the cos(2*theta)/sin(2*theta) components of
// the azimuthal anisotropy tensor, and theta is the fast-axis azimuth itself (it
// is rotated, but only on the automatic g0/theta columns).
inline const std::array<const char*, 3> rotate_model_blocked_fields = {"gc", "gs", "theta"};

// Comma-separated list of rotate_model_fields, for help text and error messages.
inline std::string rotate_model_fields_str() {
    std::string s;
    for (const auto* k : rotate_model_fields) {
        if (!s.empty()) s += ", ";
        s += k;
    }
    return s;
}

// Reduce a dataset name to its bare field name by stripping the "model_"/"grad_"
// prefix and the "_NNN" iteration suffix that model_iter.h5 uses, so that e.g.
// "model_vs_042" and "grad_gamma_007" validate as "vs" and "gamma".
inline std::string rotate_model_base_field(std::string key) {
    for (const std::string prefix : {"model_", "grad_"}) {
        if (key.rfind(prefix, 0) == 0) { key.erase(0, prefix.size()); break; }
    }
    const auto us = key.rfind('_');
    if (us != std::string::npos && us + 1 < key.size() &&
        std::all_of(key.begin() + us + 1, key.end(),
                    [](unsigned char c) { return std::isdigit(c) != 0; })) {
        key.erase(us);
    }
    return key;
}

inline RotateModelArgs argparse_rotate_model(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: surfatt_rotate_model -i model_file -o out_file -c clat/clon [-a angle] [-k keyname] [-h]\n\n"
            "Rotate a model from the local (rotated) frame back to geographic\n"
            "coordinates and convert to csv format.\n\n"
            "required arguments:\n"
            "  -i model_file        Path to model file in HDF5 format\n"
            "  -o out_file          Output csv file name\n"
            "  -c clat/clon         Centre of rotation (lat/lon)\n\n"
            "optional arguments:\n"
            "  -a angle             Rotation angle in degrees (default: 0)\n"
            "  -k keyname           Export only this scalar dataset as the model column\n"
            "                       One of: " << rotate_model_fields_str() << "\n"
            "                       optionally with a model_/grad_ prefix and an _NNN\n"
            "                       iteration suffix, e.g. model_vs_042 in model_iter.h5\n"
            "                       The csv column is named after the bare field, so\n"
            "                       model_vs_042 is written as a column named vs\n"
            "                       Without -k, vs is exported together with whichever\n"
            "                       anisotropy fields the file carries: g0/theta for\n"
            "                       azimuthal, vsv/zeta for radial\n"
            "  -h                   Print help message\n";
        std::exit(0);
    }
    RotateModelArgs out;
    out.fname    = al.require("-i");
    out.outfname = al.require("-o");
    if (auto v = al.get("-a")) out.angle   = std::stod(*v);
    if (auto v = al.get("-c")) out.center  = parse_2double(*v);
    if (auto v = al.get("-k")) {
        const std::string base = rotate_model_base_field(*v);
        const auto matches = [&base](const char* k) { return base == k; };
        // Reported directly instead of thrown: an uncaught exception under mpirun
        // buries the message in an abort trace, and this is a routine user error.
        if (std::any_of(rotate_model_blocked_fields.begin(),
                        rotate_model_blocked_fields.end(), matches)) {
            std::cerr << "Error: -k " << *v << " is not supported: \"" << base
                      << "\" is not invariant under a rotation of the frame.\n";
            if (base == "theta")
                std::cerr << "theta is the fast-axis azimuth; it is rotated together with the grid "
                             "and is\nexported next to g0 when -k is omitted and both are in the "
                             "file.\n";
            else
                std::cerr << "gc/gs are the cos(2*theta)/sin(2*theta) components of the azimuthal "
                             "anisotropy\ntensor; the derived g0/theta pair is exported when -k is "
                             "omitted.\n";
            std::exit(1);
        }
        if (!std::any_of(rotate_model_fields.begin(), rotate_model_fields.end(), matches)) {
            std::cerr << "Error: unknown dataset \"" << *v << "\" for -k.\n"
                      << "Allowed fields: " << rotate_model_fields_str() << "\n"
                      << "optionally with a model_/grad_ prefix and an _NNN iteration suffix, "
                         "e.g. model_vs_042\n";
            std::exit(1);
        }
        out.key = *v;
    }
    return out;
}

inline Tomo2DArgs argparse_tomo2d(int argc, char* argv[]) {
    ArgList al(argc, argv);
    if (al.empty() || al.has("-h")) {
        std::cout <<
            "Usage: surfatt_tomo2d -i para_file [-f] [-h]"
            " [-n nx/ny] [-p pert_vel] [-m margin_km]\n\n"
            "Adjoint-state travel time tomography for surface wave (2-D)\n\n"
            "required arguments:\n"
            "  -i para_file        Path to parameter file in yaml format\n\n"
            "optional arguments:\n"
            "  -f                  Forward simulate instead of inversion (default: false)\n"
            "  -m margin_km        Margin between anomalies in km (default: 0)\n"
            "  -n nx/ny            Number of anomalies along X, Y\n"
            "  -p pert_vel         Magnitude of velocity perturbations (default: 0.08)\n"
            "  -h                  Print help message\n";
        std::exit(0);
    }
    Tomo2DArgs out;
    out.fname  = al.require("-i");
    out.isfwd  = al.has("-f");
    if (auto v = al.get("-n")) out.ncb      = parse_2int(*v);
    if (auto v = al.get("-p")) out.pert_vel = std::stod(*v);
    if (auto v = al.get("-m")) out.hmarg    = std::stod(*v);
    return out;
}

