#include "xdmf.h"
#include "config.h"
#include "input_params.h"
#include "parallel.h"

#include <fmt/format.h>
#include <fstream>

namespace xdmf {

void write_model_iter(const std::string &xdmf_path, int iter) {
    if (!Parallel::mpi().is_main()) return;

    auto &IP = InputParams::IP();
    const std::string h5ref = MODEL_ITER_FNAME; // relative path: same directory as xdmf_path
    const int nx = ngrid_i, ny = ngrid_j, nz = ngrid_k;
    const std::string dims = fmt::format("{} {} {}", nx, ny, nz);
    const std::string di_hdr = fmt::format(
        "Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\"", dims);

    auto attr = [&](const std::string &name, const std::string &ds) {
        return fmt::format(
            "        <Attribute Name=\"{}\" AttributeType=\"Scalar\" Center=\"Node\">\n"
            "          <DataItem {}>{}/{}</DataItem>\n"
            "        </Attribute>\n",
            name, di_hdr, h5ref, ds);
    };

    std::ofstream out(xdmf_path);
    out << "<?xml version=\"1.0\" ?>\n"
           "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
           "<Xdmf xmlns:xi=\"http://www.w3.org/2001/XInclude\" Version=\"2.0\">\n"
           "  <Domain>\n"
           "    <Grid Name=\"ModelIterations\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

    for (int it = 0; it <= iter; ++it) {
        const std::string sfx = fmt::format("_{:03d}", it);
        out << fmt::format("      <Grid Name=\"iter_{:03d}\" GridType=\"Uniform\">\n", it)
            << fmt::format("        <Time Value=\"{}\"/>\n", it)
            << fmt::format("        <Topology TopologyType=\"3DRectMesh\" Dimensions=\"{}\"/>\n", dims)
            << "        <Geometry GeometryType=\"VXVYVZ\">\n"
            << fmt::format("          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">{}/x</DataItem>\n", nx, h5ref)
            << fmt::format("          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">{}/y</DataItem>\n", ny, h5ref)
            << fmt::format("          <DataItem Dimensions=\"{}\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">{}/z</DataItem>\n", nz, h5ref)
            << "        </Geometry>\n"
            << attr("vs", "model_vs" + sfx);
        if (IP.inversion().use_alpha_beta_rho) {
            out << attr("vp",  "model_vp"  + sfx)
                << attr("rho", "model_rho" + sfx);
        }
        if (IP.inversion().model_para_type == MODEL_AZI_ANI) {
            out << attr("gc", "model_gc" + sfx)
                << attr("gs", "model_gs" + sfx);
        }
        out << "      </Grid>\n";
    }

    out << "    </Grid>\n"
           "  </Domain>\n"
           "</Xdmf>\n";
}

} // namespace xdmf
