#define SIMPLE_SPRNG
#define USE_MPI
#include "../sprng5/include/sprng.h"
#define SEED 985456376

extern "C" {
    double get_sprng_random() {
        return get_rn_dbl_simple();
    }

    int get_sprng_random_int() {
        return get_rn_int_simple();
    }
}
