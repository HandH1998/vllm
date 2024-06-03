#pragma once

namespace vllm {
static inline __device__ int8_t float_to_int8_rn(float x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t &>(dst);
}
} // namespace vllm