#pragma once
#include "utils/vec_math.h"
#include <vector>
#include <iostream>
#include <array>
#include <memory>
#include "types.h"
#include "../structure.h"

namespace gsrt::kd_tracer::host::builder {

    void build(ASData_Host&, const KdParams& params);

}