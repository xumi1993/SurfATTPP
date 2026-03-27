#include "input_params.h"
#include "parallel.h"
#include "src_rec.h"

#include <algorithm>


int main(int argc, char* argv[])
{
    auto& par = Parallel::mpi();
    par.init();

    SrcRec::SR().load("data/surfker_obs.csv");

    par.finalize();
    return 0;
}