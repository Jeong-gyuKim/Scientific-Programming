#include "../sprng5/include/sprng.h"
#define SEED 985456376

extern "C" {
    void init_sprng_stream() {
        int* stream = init_rng_simple(DEFAULT_RNG_TYPE, SEED, SPRNG_DEFAULT);
        print_rng_simple();
    }

    double get_sprng_random() {
        return get_rn_dbl_simple();
    }

    int get_sprng_random_int() {
        return get_rn_int_simple();
    }
}
