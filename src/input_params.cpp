#include "input_params.h"

#include <sstream>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static std::vector<std::string> split_key(const std::string &key) {
    std::vector<std::string> parts;
    std::istringstream ss(key);
    std::string token;
    while (std::getline(ss, token, '.')) {
        if (!token.empty()) parts.push_back(token);
    }
    return parts;
}

template<typename T>
static T req(const YAML::Node &n, const std::string &field) {
    if (!n[field] || n[field].IsNull())
        throw std::runtime_error("InputParams: required field '" + field + "' is missing");
    return n[field].as<T>();
}

template<typename T>
static T opt(const YAML::Node &n, const std::string &field, T def) {
    if (!n[field] || n[field].IsNull()) return def;
    return n[field].as<T>();
}

// ---------------------------------------------------------------------------
// Section loaders
// ---------------------------------------------------------------------------

void InputParams::load_data(const YAML::Node &n) {
    data_.src_rec_file_ph = req<std::string>(n, "src_rec_file_ph");
    data_.src_rec_file_gr = opt<std::string>(n, "src_rec_file_gr", "");
    data_.iwave           = req<int>(n, "iwave");
    data_.vel_type        = req<std::vector<bool>>(n, "vel_type");
    data_.weights         = req<std::vector<real_t>>(n, "weights");
}

void InputParams::load_output(const YAML::Node &n) {
    output_.output_path   = req<std::string>(n, "output_path");
    output_.verbose_level = req<int>(n, "verbose_level");
    output_.log_level     = req<int>(n, "log_level");
}

void InputParams::load_domain(const YAML::Node &n) {
    domain_.depth           = req<std::vector<real_t>>(n, "depth");
    domain_.interval        = req<std::vector<real_t>>(n, "interval");
    domain_.num_grid_margin = req<int>(n, "num_grid_margin");
}

void InputParams::load_topo(const YAML::Node &n) {
    topo_.is_consider_topo = req<bool>(n, "is_consider_topo");
    topo_.topo_file        = opt<std::string>(n, "topo_file", "");
    topo_.wavelen_factor   = req<real_t>(n, "wavelen_factor");
}

void InputParams::load_inversion(const YAML::Node &n) {
    inversion_.use_alpha_beta_rho = req<bool>(n, "use_alpha_beta_rho");
    inversion_.rho_scaling        = req<bool>(n, "rho_scaling");

    inversion_.init_model_type = req<int>(n, "init_model_type");
    inversion_.vel_range       = req<std::vector<real_t>>(n, "vel_range");
    inversion_.init_model_path = opt<std::string>(n, "init_model_path", "");

    inversion_.kdensity_coe = req<real_t>(n, "kdensity_coe");
    inversion_.ncomponents  = req<int>(n, "ncomponents");
    inversion_.n_inv_grid   = req<std::vector<int>>(n, "n_inv_grid");

    inversion_.niter    = req<int>(n, "niter");
    inversion_.min_derr = req<real_t>(n, "min_derr");

    inversion_.optim_method  = req<int>(n, "optim_method");
    inversion_.step_length   = req<real_t>(n, "step_length");
    inversion_.maxshrink     = req<real_t>(n, "maxshrink");
    inversion_.max_sub_niter = req<int>(n, "max_sub_niter");
}

// ---------------------------------------------------------------------------
// InputParams public interface
// ---------------------------------------------------------------------------

InputParams::InputParams(const std::string &filepath)
    : filepath_(filepath) {
    try {
        root_ = YAML::LoadFile(filepath);
    } catch (const YAML::Exception &e) {
        throw std::runtime_error(
            "InputParams: failed to load '" + filepath + "': " + e.what());
    }

    auto require_section = [&](const std::string &s) -> YAML::Node {
        if (!root_[s])
            throw std::runtime_error("InputParams: missing section '" + s + "'");
        return root_[s];
    };

    load_data(require_section("data"));
    load_output(require_section("output"));
    load_domain(require_section("domain"));
    load_topo(require_section("topo"));
    load_inversion(require_section("inversion"));
}

YAML::Node InputParams::resolve(const std::string &key) const {
    auto parts = split_key(key);
    YAML::Node node = YAML::Clone(root_);
    for (const auto &part : parts) {
        if (!node.IsMap() || !node[part]) {
            return YAML::Node(YAML::NodeType::Undefined);
        }
        node = node[part];
    }
    return node;
}

bool InputParams::has(const std::string &key) const {
    YAML::Node node = resolve(key);
    return node && !node.IsNull();
}

// ---------------------------------------------------------------------------
// Broadcast helpers
// Each function mirrors the layout of its load_* counterpart.
// ---------------------------------------------------------------------------

// Convenience: resize a vector on non-main ranks then broadcast its data.
void InputParams::bcast_data() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(data_.src_rec_file_ph);
    mpi.bcast(data_.src_rec_file_gr);
    mpi.bcast(data_.iwave);
    mpi.bcast_bool_vec(data_.vel_type);
    mpi.bcast_vec(data_.weights);
}

void InputParams::bcast_output() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(output_.output_path);
    mpi.bcast(output_.verbose_level);
    mpi.bcast(output_.log_level);
}

void InputParams::bcast_domain() {
    auto &mpi = Parallel::mpi();
    mpi.bcast_vec(domain_.depth);
    mpi.bcast_vec(domain_.interval);
    mpi.bcast(domain_.num_grid_margin);
}

void InputParams::bcast_topo() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(topo_.is_consider_topo);
    mpi.bcast(topo_.topo_file);
    mpi.bcast(topo_.wavelen_factor);
}

void InputParams::bcast_inversion() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(inversion_.use_alpha_beta_rho);
    mpi.bcast(inversion_.rho_scaling);
    mpi.bcast(inversion_.init_model_type);
    mpi.bcast_vec(inversion_.vel_range);
    mpi.bcast(inversion_.init_model_path);
    mpi.bcast(inversion_.kdensity_coe);
    mpi.bcast(inversion_.ncomponents);
    mpi.bcast_vec(inversion_.n_inv_grid);
    mpi.bcast(inversion_.niter);
    mpi.bcast(inversion_.min_derr);
    mpi.bcast(inversion_.optim_method);
    mpi.bcast(inversion_.step_length);
    mpi.bcast(inversion_.maxshrink);
    mpi.bcast(inversion_.max_sub_niter);
}

void InputParams::bcast_all_params() {
    bcast_data();
    bcast_output();
    bcast_domain();
    bcast_topo();
    bcast_inversion();
}

