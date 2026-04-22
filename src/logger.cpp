#include "logger.h"
#include "parallel.h"

// ---------------------------------------------------------------------------
// Singleton
// ---------------------------------------------------------------------------

void ATTLogger::init(const std::string &logger_file, int log_level, bool console_only) {
    get_instance_ptr() =
        std::unique_ptr<ATTLogger>(new ATTLogger(logger_file, log_level, console_only));
    get_instance_ptr() -> Info(
        "------------------------------------------------------", MODULE_MAIN);
    get_instance_ptr() -> Info(
        fmt::format("----------  SurfATT {}, commit: {}.  ---------", get_version_number(), GIT_COMMIT), MODULE_MAIN);
    get_instance_ptr() -> Info(
        "------------------------------------------------------", MODULE_MAIN);
}

ATTLogger &ATTLogger::logger() {
    auto *ptr = get_instance_ptr().get();
    if (!ptr) throw std::runtime_error("ATTLogger: call init() first");
    return *ptr;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ATTLogger::ATTLogger(const std::string &logger_file, int log_level, bool console_only) {
    auto console_sink =
        std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    if (!console_only) {
        auto file_sink =
            std::make_shared<spdlog::sinks::basic_file_sink_mt>(logger_file, true);
        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        spdlogger_ = std::make_shared<spdlog::logger>("multi_sink",
                                                       begin(sinks), end(sinks));
    } else {
        spdlogger_ = std::make_shared<spdlog::logger>("console_logger",
                                                       console_sink);
    }

    auto level = static_cast<spdlog::level::level_enum>(log_level);
    spdlogger_->flush_on(level);
    spdlogger_->set_level(level);
    spdlog::set_default_logger(spdlogger_);
}

// ---------------------------------------------------------------------------
// Logging methods
// ---------------------------------------------------------------------------

void ATTLogger::Debug(const std::string &msg, const std::string &module_name,
                      bool main_only) {
    if (Parallel::mpi().is_main() || !main_only) {
        spdlogger_->set_pattern(set_format(module_name));
        spdlogger_->debug(msg);
    }
}

void ATTLogger::Info(const std::string &msg, const std::string &module_name,
                     bool main_only) {
    if (Parallel::mpi().is_main() || !main_only) {
        spdlogger_->set_pattern(set_format(module_name));
        spdlogger_->info(msg);
    }
}

void ATTLogger::Warn(const std::string &msg, const std::string &module_name) {
    if (Parallel::mpi().is_main()) {
        spdlogger_->set_pattern(set_format(module_name));
        spdlogger_->warn(msg);
    }
}

void ATTLogger::Error(const std::string &msg, const std::string &module_name) {
    if (Parallel::mpi().is_main()) {
        spdlogger_->set_pattern(set_format(module_name));
        spdlogger_->error(msg);
    }
}

void ATTLogger::shutdown() {
    Info("------------------------------------------------------", MODULE_MAIN);
    Info("------------  SurfATT Calculation Down.  -------------", MODULE_MAIN);
    Info("------------------------------------------------------", MODULE_MAIN);
    spdlog::shutdown();
}