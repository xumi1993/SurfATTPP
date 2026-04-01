#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <stdexcept>
#include "parallel.h"
#include "config.h"

// ---------------------------------------------------------------------------
// Typed parameter structs — one per top-level YAML section
// ---------------------------------------------------------------------------

struct DataParams {
    std::string src_rec_file_ph;   // source-receiver file for phase velocity
    std::string src_rec_file_gr;   // source-receiver file for group velocity (optional)
    int         iwave;             // 1 = Love, 2 = Rayleigh
    std::vector<bool>   vel_type;  // [use_phase, use_group]
    std::vector<real_t> weights;   // [weight_phase, weight_group]
};

struct OutputParams {
    std::string output_path;
    bool output_in_process_data;
    bool output_initial_model;
    int log_level;      // 0 debug, 1 verbose
};

struct DomainParams {
    std::vector<real_t> depth;           // [min_km, max_km]
    std::vector<real_t> interval;        // [dlon_deg, dlat_deg, dz_km]
    int                 num_grid_margin;
};

struct TopoParams {
    bool        is_consider_topo;
    std::string topo_file;
    real_t      wavelen_factor;
};

struct InversionParams {
    // model parametrisation
    bool is_anisotropy;
    bool use_alpha_beta_rho;
    bool rho_scaling;
    // initial model
    int                 init_model_type;  // 0 linear, 1 1-D avg, 2 external 3-D
    std::vector<real_t> vel_range;        // [v_min, v_max] km/s
    std::string         init_model_path;  // only used when init_model_type == 2
    // regularisation
    real_t kdensity_coe;
    bool is_kden;
    int    ncomponents;
    std::vector<int> n_inv_grid;
    // convergence
    int    niter;
    real_t min_derr;
    // optimisation
    int    optim_method;   // 0 SD, 1 CG, 2 LBFGS
    real_t step_length;
    real_t maxshrink;
    int    max_sub_niter;
};

// ---------------------------------------------------------------------------
// InputParams — loads an input_params.yml file
// ---------------------------------------------------------------------------
class InputParams {
public:
    // Singleton access -------------------------------------------------------
    // Call once at program startup to load the file, then call IP()
    // from anywhere to access the parameters.
    //   InputParams::read("input_params.yml");
    //   auto &inv = InputParams::IP().inversion();
    // Only rank 0 needs the file; all other ranks pass an empty string.
    // After construction, bcast() is called automatically.
    static void read(const std::string &filepath) {
        auto &mpi = Parallel::mpi();
        // Only the main rank loads the file; others construct with empty path
        // and receive data via bcast().
        if (mpi.is_main()) {
            get_IP_ptr() = std::make_unique<InputParams>(filepath);
        } else {
            get_IP_ptr() = std::make_unique<InputParams>();
        }
        get_IP_ptr()->bcast_all_params();
    }
    static InputParams &IP() {
        auto *ptr = get_IP_ptr().get();
        if (!ptr) throw std::runtime_error("InputParams: call read() first");
        return *ptr;
    }

    // Load YAML file (rank 0 only). Throws std::runtime_error on failure.
    explicit InputParams(const std::string &filepath);
    // Default constructor for non-main ranks (fields filled by bcast).
    InputParams() = default;

    // Broadcast all parameters from rank 0 to all other ranks.
    void bcast_all_params();
    // Typed accessors for each section
    const DataParams      &data()      const { return data_; }
    const OutputParams    &output()    const { return output_; }
    const DomainParams    &domain()    const { return domain_; }
    const TopoParams      &topo()      const { return topo_; }
    const InversionParams &inversion() const { return inversion_; }

    // Generic dot-notation accessor, e.g. get<int>("inversion.niter")
    template<typename T>
    T get(const std::string &key) const {
        YAML::Node node = resolve(key);
        if (!node) {
            throw std::runtime_error("InputParams: missing key '" + key + "'");
        }
        return node.as<T>();
    }

    template<typename T>
    T get(const std::string &key, const T &default_val) const {
        YAML::Node node = resolve(key);
        if (!node) return default_val;
        return node.as<T>();
    }

    bool has(const std::string &key) const;

    const YAML::Node &root() const { return root_; }

private:
    static std::unique_ptr<InputParams> &get_IP_ptr() {
        static std::unique_ptr<InputParams> IP;
        return IP;
    }

    YAML::Node      root_;
    std::string     filepath_;

    DataParams      data_;
    OutputParams    output_;
    DomainParams    domain_;
    TopoParams      topo_;
    InversionParams inversion_;

    YAML::Node resolve(const std::string &key) const;

    void load_data(const YAML::Node &n);
    void load_output(const YAML::Node &n);
    void load_domain(const YAML::Node &n);
    void load_topo(const YAML::Node &n);
    void load_inversion(const YAML::Node &n);

    void bcast_data();
    void bcast_output();
    void bcast_domain();
    void bcast_topo();
    void bcast_inversion();
};
