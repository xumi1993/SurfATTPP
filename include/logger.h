#pragma once

#include "config.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <memory>

class ATTLogger {
public:
    // Singleton access -------------------------------------------------------
    // Call once at startup:  ATTLogger::init(LOG_FNAME);
    // Then from anywhere:    ATTLogger::logger().Info(...);
    static void       init(const std::string &logger_file,
                           int  log_level    = 1,
                           bool console_only = false);
    static ATTLogger &logger();

    // Logging interface
    void Debug(const std::string&, const std::string&, bool main_only = true);
    void Info (const std::string&, const std::string&, bool main_only = true);
    void Warn (const std::string&, const std::string&);
    void Error(const std::string&, const std::string&);
    void shutdown();

private:
    explicit ATTLogger(const std::string &logger_file, int log_level, bool console_only);

    static std::unique_ptr<ATTLogger> &get_instance_ptr() {
        static std::unique_ptr<ATTLogger> inst;
        return inst;
    }

    std::shared_ptr<spdlog::logger> spdlogger_;

    inline std::string set_format(const std::string &module_name) {
        std::string upper = module_name;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        return "[%Y-%m-%d %H:%M:%S.%e] [" + upper + "] [%^%l%$] %v";
    }
};

