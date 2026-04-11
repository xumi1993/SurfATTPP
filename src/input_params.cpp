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
    output_.output_path   = opt<std::string>(n, "output_path", "OUTPUT_FILES/");
    output_.output_in_process_data = opt<bool>(n, "output_in_process_data", false);
    output_.output_initial_model = opt<bool>(n, "output_initial_model", false);
    output_.output_in_process_model = opt<bool>(n, "output_in_process_model", false);
    output_.log_level     = opt<int>(n, "log_level", 1);
}

void InputParams::load_domain(const YAML::Node &n) {
    domain_.grid_method     = opt<int>(n, "grid_method", 0);
    domain_.depth           = req<std::vector<real_t>>(n, "depth_min_max");
    if (domain_.grid_method == 0) {
        YAML::Node ng = n["grid_method_0"];
        // Require depth/interval, compute n_grid from them
        domain_.interval        = req<std::vector<real_t>>(ng, "interval");
        domain_.num_grid_margin = opt<int>(ng, "num_grid_margin", 5);
    } else if (domain_.grid_method == 1) {
        YAML::Node ng = n["grid_method_1"];
        // Require lat/lon min/max and number of points
        domain_.n_grid          = req<std::vector<int>>(ng, "n_grid");
        domain_.lat_min_max     = req<std::vector<real_t>>(ng, "lat_min_max");
        domain_.lon_min_max     = req<std::vector<real_t>>(ng, "lon_min_max");
    } else {
        throw std::runtime_error("InputParams: unsupported grid_method " + std::to_string(domain_.grid_method));
    }
}

void InputParams::load_model(const YAML::Node &n) {
    model_.init_model_type = req<int>(n, "init_model_type");
    model_.vel_range       = req<std::vector<real_t>>(n, "vel_range");
    model_.init_model_path = opt<std::string>(n, "init_model_path", "");
}

void InputParams::load_topo(const YAML::Node &n) {
    topo_.is_consider_topo = req<bool>(n, "is_consider_topo");
    topo_.topo_file        = opt<std::string>(n, "topo_file", "");
    topo_.wavelen_factor   = req<real_t>(n, "wavelen_factor");
}

void InputParams::load_postproc(const YAML::Node &n) {
    postproc_.kdensity_coe = opt<real_t>(n, "kdensity_coe", _0_CR);
    postproc_.is_kden = std::abs(postproc_.kdensity_coe - _0_CR) > 1e-6;
    postproc_.smooth_method = opt<int>(n, "smooth_method", 0);
    postproc_.independent_smooth_ani = opt<bool>(n, "independent_smooth_ani", false);

    // Support hierarchical config:
    // postproc:
    //   smooth_method: 0/1
    //   smooth_method_0: {sigma, sigma_ani}
    //   smooth_method_1: {n_inv_components, n_inv_grid, n_inv_grid_ani}
    //
    // Keep backward compatibility with the old flat fields.
    if (postproc_.smooth_method == 0) {
        const YAML::Node sm = n["smooth_method_0"];
        postproc_.sigma = req<std::vector<real_t>>(sm, "sigma");
        if (postproc_.independent_smooth_ani) {
            postproc_.sigma_ani = req<std::vector<real_t>>(sm, "sigma_ani");
        } else {
            postproc_.sigma_ani = postproc_.sigma;
        }
    } else if (postproc_.smooth_method == 1) {
        const YAML::Node sm = n["smooth_method_1"];
        postproc_.n_inv_components = req<int>(sm, "n_inv_components");
        postproc_.n_inv_grid = req<std::vector<int>>(sm, "n_inv_grid");
        if (postproc_.independent_smooth_ani) {
            postproc_.n_inv_grid_ani = req<std::vector<int>>(sm, "n_inv_grid_ani");
        } else {
            postproc_.n_inv_grid_ani = postproc_.n_inv_grid;
        }
    } else {
         throw std::runtime_error("InputParams: unsupported smooth_method " + std::to_string(postproc_.smooth_method));
    }
}

void InputParams::load_inversion(const YAML::Node &n) {
    inversion_.is_anisotropy      = opt<bool>(n, "is_anisotropy", false);
    inversion_.use_alpha_beta_rho = req<bool>(n, "use_alpha_beta_rho");
    inversion_.rho_scaling        = req<bool>(n, "rho_scaling");

    inversion_.niter    = req<int>(n, "niter");
    inversion_.min_derr = req<real_t>(n, "min_derr");

    inversion_.optim_method  = req<int>(n, "optim_method");
    inversion_.step_length   = req<real_t>(n, "step_length");
    inversion_.maxshrink     = req<real_t>(n, "maxshrink");
    inversion_.max_sub_niter = req<int>(n, "max_sub_niter");
    // Strong Wolfe line-search parameters. Keep YAML fallbacks consistent with
    // the defaults defined in InversionParams so behavior does not depend on
    // whether the key is omitted from the input file.
    inversion_.c1 = opt<real_t>(n, "c1", 0.1);
    inversion_.c2 = opt<real_t>(n, "c2", 0.9);
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
    load_model(require_section("model"));
    load_topo(require_section("topo"));
    load_postproc(require_section("postproc"));
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
    mpi.bcast(output_.output_in_process_data);
    mpi.bcast(output_.output_initial_model);
    mpi.bcast(output_.output_in_process_model);
    mpi.bcast(output_.log_level);
}

void InputParams::bcast_domain() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(domain_.grid_method);
    mpi.bcast_vec(domain_.depth);
    mpi.bcast_vec(domain_.interval);
    mpi.bcast(domain_.num_grid_margin);
    mpi.bcast_vec(domain_.lat_min_max);
    mpi.bcast_vec(domain_.lon_min_max);
    mpi.bcast_vec(domain_.n_grid);
}

void InputParams::bcast_model() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(model_.init_model_type);
    mpi.bcast_vec(model_.vel_range);
    mpi.bcast(model_.init_model_path);
}

void InputParams::bcast_topo() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(topo_.is_consider_topo);
    mpi.bcast(topo_.topo_file);
    mpi.bcast(topo_.wavelen_factor);
}

void InputParams::bcast_postproc() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(postproc_.kdensity_coe);
    mpi.bcast(postproc_.is_kden);
    mpi.bcast(postproc_.smooth_method);
    mpi.bcast_vec(postproc_.sigma);
    mpi.bcast_vec(postproc_.sigma_ani);
    mpi.bcast(postproc_.n_inv_components);
    mpi.bcast_vec(postproc_.n_inv_grid);
    mpi.bcast_vec(postproc_.n_inv_grid_ani);
    mpi.bcast(postproc_.independent_smooth_ani);
}

void InputParams::bcast_inversion() {
    auto &mpi = Parallel::mpi();
    mpi.bcast(inversion_.is_anisotropy);
    mpi.bcast(inversion_.use_alpha_beta_rho);
    mpi.bcast(inversion_.rho_scaling);
    mpi.bcast(inversion_.niter);
    mpi.bcast(inversion_.min_derr);
    mpi.bcast(inversion_.optim_method);
    mpi.bcast(inversion_.step_length);
    mpi.bcast(inversion_.maxshrink);
    mpi.bcast(inversion_.max_sub_niter);
    mpi.bcast(inversion_.c1);
    mpi.bcast(inversion_.c2);
}

void InputParams::bcast_all_params() {
    bcast_data();
    bcast_output();
    bcast_domain();
    bcast_model();
    bcast_topo();
    bcast_postproc();
    bcast_inversion();
}
