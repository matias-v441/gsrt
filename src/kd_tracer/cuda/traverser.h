#pragma once
#include "../structure.h"

namespace gsrt::kd_tracer::cuda::traverser {

	constexpr int HIT_BUFFER_SIZE = 4096;

	void rcast_kd_restart(const ASData_Device&, const RayData&, const RenderOutput&);
	
}