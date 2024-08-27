#include "./util.h"
#include <cstdint>
#include <random>

namespace mevil {

namespace {
    static std::random_device rd;
    static std::mt19937 gen(rd());
}

std::uint64_t random() {
    return gen();
}

}