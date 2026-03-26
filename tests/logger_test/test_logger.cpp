#include "logger.h"
#include "parallel.h"
#include "config.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

static void print(const std::string &msg) {
    if (Parallel::mpi().is_main()) std::cout << msg << "\n";
}

// ---------------------------------------------------------------------------
// Test: calling logger() before init() must throw
// ---------------------------------------------------------------------------
static void test_throws_before_init() {
    bool threw = false;
    try {
        ATTLogger::logger();
    } catch (const std::runtime_error &) {
        threw = true;
    }
    assert(threw && "logger() must throw before init()");
    print("[PASS] throws before init");
}

// ---------------------------------------------------------------------------
// Test: console-only mode — no log file created
// ---------------------------------------------------------------------------
static void test_console_only() {
    const std::string log_file = "test_console_only.log";
    ATTLogger::init(log_file, /*log_level=*/0, /*console_only=*/true);

    auto &log = ATTLogger::logger();
    log.Debug("debug message",   "TestModule");
    log.Info ("info  message",   "TestModule");
    log.Warn ("warn  message",   "TestModule");
    log.Error("error message",   "TestModule");
    log.shutdown();

    // No file should have been created
    assert(!fs::exists(log_file) && "console_only must not create a log file");
    print("[PASS] console_only (no file created)");
}

// ---------------------------------------------------------------------------
// Test: file mode — log file is created and non-empty
// ---------------------------------------------------------------------------
static void test_file_output() {
    const std::string log_file = "test_logger_output.log";
    // Remove possible leftover from a previous run
    fs::remove(log_file);

    // log_level 0 = trace → all messages visible
    ATTLogger::init(log_file, /*log_level=*/0, /*console_only=*/false);

    auto &log = ATTLogger::logger();
    log.Debug("debug entry",   "FileTest");
    log.Info ("info  entry",   "FileTest");
    log.Warn ("warn  entry",   "FileTest");
    log.Error("error entry",   "FileTest");

    // main_only=false: non-main ranks also write
    log.Info("rank message", "FileTest", /*main_only=*/false);
    log.shutdown();

    // File must exist and contain at least one line
    if (Parallel::mpi().is_main()) {
        assert(fs::exists(log_file) && "log file must be created in file mode");
        std::ifstream f(log_file);
        std::string line;
        bool has_content = static_cast<bool>(std::getline(f, line));
        assert(has_content && "log file must not be empty");
        fs::remove(log_file);    // clean up
    }
    Parallel::mpi().barrier();
    print("[PASS] file output (created and non-empty)");
}

// ---------------------------------------------------------------------------
// Test: log_level filtering
//   level 3 = warn → debug and info should be suppressed (no crash, silent)
// ---------------------------------------------------------------------------
static void test_log_level_filtering() {
    const std::string log_file = "test_logger_level.log";
    fs::remove(log_file);

    // level 3 = warn
    ATTLogger::init(log_file, /*log_level=*/3, /*console_only=*/false);

    auto &log = ATTLogger::logger();
    // These should silently be filtered out by spdlog
    log.Debug("should be suppressed", "LevelTest");
    log.Info ("should be suppressed", "LevelTest");
    // These must go through
    log.Warn ("warn passes",  "LevelTest");
    log.Error("error passes", "LevelTest");
    log.shutdown();

    if (Parallel::mpi().is_main()) {
        assert(fs::exists(log_file));
        // Count lines: only warn + error should appear (2 lines)
        std::ifstream f(log_file);
        int lines = 0;
        std::string line;
        while (std::getline(f, line)) ++lines;
        assert(lines == 2 && "only warn+error should be written at level 3");
        fs::remove(log_file);
    }
    Parallel::mpi().barrier();
    print("[PASS] log_level filtering (warn+error only at level 3)");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    Parallel::init();

    print("=== ATTLogger tests ===");

    test_throws_before_init();
    test_console_only();
    test_file_output();
    test_log_level_filtering();

    print("=== All logger tests passed ===");

    Parallel::mpi().finalize();
    return 0;
}
