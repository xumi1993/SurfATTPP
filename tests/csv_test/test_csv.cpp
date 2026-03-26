#include "rapidcsv.h"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void assert_near(double a, double b, double tol = 1e-9) {
    assert(std::abs(a - b) < tol);
}

// ---------------------------------------------------------------------------
// Test: write a CSV file then read it back
// ---------------------------------------------------------------------------
static void test_write_read() {
    const std::string path = "test_write_read.csv";
    fs::remove(path);

    // --- Write ---
    rapidcsv::Document doc;
    doc.SetColumn<std::string>(0, {"Alice", "Bob", "Carol"});
    doc.SetColumn<int>        (1, {30, 25, 28});
    doc.SetColumn<double>     (2, {1.70, 1.82, 1.65});
    doc.SetColumnName(0, "name");
    doc.SetColumnName(1, "age");
    doc.SetColumnName(2, "height");
    doc.Save(path);
    assert(fs::exists(path));

    // --- Read back ---
    rapidcsv::Document in(path);

    // column by name
    auto names   = in.GetColumn<std::string>("name");
    auto ages    = in.GetColumn<int>        ("age");
    auto heights = in.GetColumn<double>     ("height");

    assert(names.size()   == 3);
    assert(names[0] == "Alice" && names[1] == "Bob" && names[2] == "Carol");
    assert(ages[0]  == 30      && ages[1]  == 25    && ages[2]  == 28);
    assert_near(heights[0], 1.70);
    assert_near(heights[1], 1.82);
    assert_near(heights[2], 1.65);

    fs::remove(path);
    std::cout << "[PASS] write + read (column by name)\n";
}

// ---------------------------------------------------------------------------
// Test: read by row index
// ---------------------------------------------------------------------------
static void test_read_by_row() {
    const std::string path = "test_read_by_row.csv";
    fs::remove(path);

    rapidcsv::Document doc;
    doc.SetColumn<int>(0, {10, 20, 30});
    doc.SetColumn<int>(1, {11, 21, 31});
    doc.SetColumnName(0, "A");
    doc.SetColumnName(1, "B");
    doc.Save(path);

    rapidcsv::Document in(path);
    // row 0 → {10, 11}
    auto row0 = in.GetRow<int>(0);
    assert(row0.size() == 2);
    assert(row0[0] == 10 && row0[1] == 11);
    // row 2 → {30, 31}
    auto row2 = in.GetRow<int>(2);
    assert(row2[0] == 30 && row2[1] == 31);

    assert(in.GetRowCount() == 3);
    assert(in.GetColumnCount() == 2);

    fs::remove(path);
    std::cout << "[PASS] read by row index\n";
}

// ---------------------------------------------------------------------------
// Test: modify a cell and re-save
// ---------------------------------------------------------------------------
static void test_modify_cell() {
    const std::string path = "test_modify_cell.csv";
    fs::remove(path);

    rapidcsv::Document doc;
    doc.SetColumn<int>(0, {1, 2, 3});
    doc.SetColumnName(0, "v");
    doc.Save(path);

    rapidcsv::Document in(path);
    // overwrite row 1 value
    in.SetCell<int>(0, 1, 99);
    in.Save(path);

    rapidcsv::Document in2(path);
    auto col = in2.GetColumn<int>("v");
    assert(col[0] == 1 && col[1] == 99 && col[2] == 3);

    fs::remove(path);
    std::cout << "[PASS] modify cell + re-save\n";
}

// ---------------------------------------------------------------------------
// Test: custom separator (tab-separated)
// ---------------------------------------------------------------------------
static void test_custom_separator() {
    const std::string path = "test_tsv.tsv";
    fs::remove(path);

    {
        std::ofstream f(path);
        f << "x\ty\n1\t2\n3\t4\n";
    }

    rapidcsv::Document in(path,
        rapidcsv::LabelParams(),
        rapidcsv::SeparatorParams('\t'));

    auto x = in.GetColumn<int>("x");
    auto y = in.GetColumn<int>("y");
    assert(x[0] == 1 && x[1] == 3);
    assert(y[0] == 2 && y[1] == 4);

    fs::remove(path);
    std::cout << "[PASS] custom separator (tab)\n";
}

// ---------------------------------------------------------------------------
// Test: no-header mode (LabelParams(-1,-1))
// ---------------------------------------------------------------------------
static void test_no_header() {
    const std::string path = "test_no_header.csv";
    fs::remove(path);

    {
        std::ofstream f(path);
        f << "10,20\n30,40\n";
    }

    rapidcsv::Document in(path,
        rapidcsv::LabelParams(-1, -1));   // no column or row labels

    auto col0 = in.GetColumn<int>(0);
    auto col1 = in.GetColumn<int>(1);
    assert(col0[0] == 10 && col0[1] == 30);
    assert(col1[0] == 20 && col1[1] == 40);

    fs::remove(path);
    std::cout << "[PASS] no-header mode\n";
}

// ---------------------------------------------------------------------------
// Test: missing file throws
// ---------------------------------------------------------------------------
static void test_missing_file_throws() {
    bool threw = false;
    try {
        rapidcsv::Document in("nonexistent_file_xyz.csv");
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw && "loading a missing file must throw");
    std::cout << "[PASS] missing file throws\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== rapidcsv tests ===\n";

    test_write_read();
    test_read_by_row();
    test_modify_cell();
    test_custom_separator();
    test_no_header();
    test_missing_file_throws();

    std::cout << "=== All CSV tests passed ===\n";
    return 0;
}
