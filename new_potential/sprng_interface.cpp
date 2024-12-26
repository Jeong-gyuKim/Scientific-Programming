#define SIMPLE_SPRNG
#define USE_MPI
#include "../sprng5/include/sprng.h"

extern "C" {
	void init_sprng_stream(int seed, int param, int rng_type) {
		init_sprng(seed, param, rng_type);
		print_sprng();
	}

    double get_sprng_random() {
        return sprng();
    }

    int get_sprng_random_int() {
        return isprng();
    }
}
