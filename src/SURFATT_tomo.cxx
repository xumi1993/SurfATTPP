#include "input_params.h"
#include "parallel.h"
#include "src_rec.h"
#include "logger.h"
#include "config.h"
#include "utils.h"

#include <algorithm>


int main(int argc, char* argv[])
{
    // parse options
    parse_options(argc, argv);

    // initialise MPI
    Parallel::init();
    auto& mpi = Parallel::mpi();

    // read input parameters
    InputParams::read(input_file);
    auto& IP = InputParams::IP();

    // initialize logger
    ATTLogger::init(
        std::format("{}/{}", IP.output().output_path, LOG_FNAME),
        IP.output().log_level,
        false
    );

    // load source-receiver tables into shared memory
    if (IP.data().vel_type[0]) SrcRec::SR_ph().load(IP.data().src_rec_file_ph);
    if (IP.data().vel_type[1]) SrcRec::SR_gr().load(IP.data().src_rec_file_gr);
    SrcRec::build_stas();

    // check data
    std::cout << "Total number of stations: " << SrcRec::stas().stnm[10] << std::endl;

    SrcRec::SR_ph().release_shm();
    SrcRec::SR_gr().release_shm();

    return 0;
}