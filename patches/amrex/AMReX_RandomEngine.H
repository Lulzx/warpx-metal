#ifndef AMREX_RANDOM_ENGINE_H_
#define AMREX_RANDOM_ENGINE_H_
#include <AMReX_Config.H>

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Extension.H>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#ifdef AMREX_USE_OMP
#  include <AMReX_BLassert.H>
#  include <omp.h>
#endif

#if defined(AMREX_USE_HIP)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#include <hiprand/hiprand.hpp>
#pragma clang diagnostic pop
#elif defined(AMREX_USE_CUDA)
#include <curand.h>
#include <curand_kernel.h>
#elif defined(AMREX_USE_SYCL)
#include <sycl/sycl.hpp>
#if !defined(SYCL_IMPLEMENTATION_ACPP) && !defined(SYCL_IMPLEMENTATION_HIPSYCL)
// Intel oneAPI MKL RNG — not available with AdaptiveCpp
#include <oneapi/mkl/rng.hpp>
#include <oneapi/mkl/rng/device.hpp>
namespace mkl = oneapi::mkl;
#endif
#endif

namespace amrex
{
#ifdef AMREX_USE_GPU

#ifdef AMREX_USE_SYCL

#if !defined(SYCL_IMPLEMENTATION_ACPP) && !defined(SYCL_IMPLEMENTATION_HIPSYCL)
    // Intel oneAPI MKL RNG types
    using sycl_rng_engine = mkl::rng::device::philox4x32x10<>;
    using sycl_rng_descr = mkl::rng::device::engine_descriptor<sycl_rng_engine>;
    using sycl_rng_acc = mkl::rng::device::engine_accessor<sycl_rng_engine>;

    extern sycl_rng_descr* rand_engine_descr;

    inline
    sycl_rng_descr* getRandEngineDescriptor () {
#ifdef AMREX_USE_OMP
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(! omp_in_parallel(), "RNG on GPU is not OpenMP thread safe");
#endif
        return rand_engine_descr;
    }

    struct RandomEngine {
        sycl_rng_engine* engine;
        AMREX_GPU_HOST_DEVICE RandomEngine (sycl_rng_engine* e) : engine(e) {}
    };
#else
    struct acpp_rng_engine {
        std::uint64_t counter;
        std::uint32_t key0;
        std::uint32_t key1;

        AMREX_GPU_HOST_DEVICE acpp_rng_engine () : counter(0), key0(0), key1(0) {}
        AMREX_GPU_HOST_DEVICE explicit acpp_rng_engine (std::uint64_t s)
            : counter(0),
              key0(static_cast<std::uint32_t>(s)),
              key1(static_cast<std::uint32_t>(s >> 32U)) {}
        AMREX_GPU_HOST_DEVICE acpp_rng_engine (std::uint64_t seed, std::uint64_t seq)
            : counter(seq),
              key0(static_cast<std::uint32_t>(seed) ^
                   static_cast<std::uint32_t>(seq * 0x9e3779b97f4a7c15ULL)),
              key1(static_cast<std::uint32_t>(seed >> 32U) ^
                   static_cast<std::uint32_t>((seq * 0xbf58476d1ce4e5b9ULL) >> 32U)) {}

        AMREX_GPU_HOST_DEVICE static void bump_key (
            std::uint32_t& k0, std::uint32_t& k1)
        {
            k0 += 0x9e3779b9U;
            k1 += 0xbb67ae85U;
        }

        AMREX_GPU_HOST_DEVICE static void philox_round (
            std::uint32_t& c0, std::uint32_t& c1,
            std::uint32_t& c2, std::uint32_t& c3,
            std::uint32_t k0, std::uint32_t k1)
        {
            std::uint64_t const p0 = std::uint64_t(0xd2511f53U) * c0;
            std::uint64_t const p1 = std::uint64_t(0xcd9e8d57U) * c2;
            std::uint32_t const hi0 = static_cast<std::uint32_t>(p0 >> 32U);
            std::uint32_t const lo0 = static_cast<std::uint32_t>(p0);
            std::uint32_t const hi1 = static_cast<std::uint32_t>(p1 >> 32U);
            std::uint32_t const lo1 = static_cast<std::uint32_t>(p1);

            std::uint32_t const n0 = hi1 ^ c1 ^ k0;
            std::uint32_t const n1 = lo1;
            std::uint32_t const n2 = hi0 ^ c3 ^ k1;
            std::uint32_t const n3 = lo0;
            c0 = n0;
            c1 = n1;
            c2 = n2;
            c3 = n3;
        }

        AMREX_GPU_HOST_DEVICE std::uint64_t next_u64 () {
            std::uint64_t const ctr = counter++;
            std::uint32_t c0 = static_cast<std::uint32_t>(ctr);
            std::uint32_t c1 = static_cast<std::uint32_t>(ctr >> 32U);
            std::uint32_t c2 = key0 ^ 0x243f6a88U;
            std::uint32_t c3 = key1 ^ 0x85a308d3U;
            std::uint32_t k0 = key0;
            std::uint32_t k1 = key1;

            for (int round = 0; round < 10; ++round) {
                philox_round(c0, c1, c2, c3, k0, k1);
                bump_key(k0, k1);
            }

            return (std::uint64_t(c0) << 32U) | std::uint64_t(c2);
        }

        template <typename RealType>
        AMREX_GPU_HOST_DEVICE RealType uniform () {
            if constexpr (std::numeric_limits<RealType>::digits <= 24) {
                return static_cast<RealType>(next_u64() >> 40U) *
                       (RealType(1.0) / RealType(16777216.0));
            } else {
                return static_cast<RealType>(next_u64() >> 11U) *
                       (RealType(1.0) / RealType(9007199254740992.0));
            }
        }

        AMREX_GPU_HOST_DEVICE unsigned int uniform_uint (unsigned int n) {
            if (n == 0) {
                return 0;
            }
            std::uint64_t const limit =
                std::numeric_limits<std::uint64_t>::max() -
                (std::numeric_limits<std::uint64_t>::max() % n);
            std::uint64_t r;
            do {
                r = next_u64();
            } while (r >= limit);
            return static_cast<unsigned int>(r % n);
        }

        template <typename RealType>
        AMREX_GPU_HOST_DEVICE RealType normal () {
            RealType u1 = uniform<RealType>();
            RealType const min_u = std::numeric_limits<RealType>::min();
            if (u1 < min_u) {
                u1 = min_u;
            }
            RealType const u2 = uniform<RealType>();
            RealType const two = RealType(2.0);
            RealType const pi = RealType(3.141592653589793238462643383279502884);
            return std::sqrt(-two * std::log(u1)) * std::cos(two * pi * u2);
        }

        template <typename RealType>
        AMREX_GPU_HOST_DEVICE unsigned int poisson (RealType lambda) {
            if (lambda <= RealType(0.0)) {
                return 0;
            }
            if (lambda < RealType(64.0)) {
                RealType const threshold = std::exp(-lambda);
                RealType product = RealType(1.0);
                unsigned int k = 0;
                do {
                    ++k;
                    product *= uniform<RealType>();
                } while (product > threshold);
                return k - 1U;
            }

            // PTRS transformed-rejection sampler (Hoermann 1993), driven only by
            // this engine's Philox uniforms. This preserves Poisson tails and
            // discreteness where the old rounded-normal branch was biased.
            RealType const slam = std::sqrt(lambda);
            RealType const loglam = std::log(lambda);
            RealType const b = RealType(0.931) + RealType(2.53) * slam;
            RealType const a = RealType(-0.059) + RealType(0.02483) * b;
            RealType const inv_alpha = RealType(1.1239) + RealType(1.1328) / (b - RealType(3.4));
            RealType const v_r = RealType(0.9277) - RealType(3.6224) / (b - RealType(2.0));

            while (true) {
                RealType const u = uniform<RealType>() - RealType(0.5);
                RealType const v = uniform<RealType>();
                RealType const us = RealType(0.5) - std::abs(u);
                if (us <= RealType(0.0) || v <= RealType(0.0)) {
                    continue;
                }

                RealType const k_real = std::floor((RealType(2.0) * a / us + b) * u + lambda + RealType(0.43));
                if (k_real < RealType(0.0)) {
                    continue;
                }
                if (us >= RealType(0.07) && v <= v_r) {
                    return static_cast<unsigned int>(k_real);
                }
                if (us < RealType(0.013) && v > us) {
                    continue;
                }

                RealType const lhs = std::log(v * inv_alpha / (a / (us * us) + b));
                RealType const rhs = -lambda + k_real * loglam - std::lgamma(k_real + RealType(1.0));
                if (lhs <= rhs) {
                    if (k_real > RealType(std::numeric_limits<unsigned int>::max())) {
                        return std::numeric_limits<unsigned int>::max();
                    }
                    return static_cast<unsigned int>(k_real);
                }
            }
        }
    };

    struct acpp_rng_acc {
        acpp_rng_engine* engines;
        std::size_t size;

        AMREX_GPU_HOST_DEVICE acpp_rng_engine load (std::size_t i) const {
            return engines[i];
        }
        AMREX_GPU_HOST_DEVICE void store (acpp_rng_engine const& engine, std::size_t i) const {
            engines[i] = engine;
        }
    };

    struct acpp_rng_descr {
        acpp_rng_engine* engines = nullptr;
        std::size_t size = 0;

        acpp_rng_descr () = default;
        acpp_rng_descr (acpp_rng_engine* p, std::size_t n) : engines(p), size(n) {}
        acpp_rng_acc get_access (sycl::handler&) const { return acpp_rng_acc{engines, size}; }
    };

    using sycl_rng_engine = acpp_rng_engine;
    using sycl_rng_descr = acpp_rng_descr;
    using sycl_rng_acc = acpp_rng_acc;

    extern sycl_rng_descr* rand_engine_descr;

    inline sycl_rng_descr* getRandEngineDescriptor () {
#ifdef AMREX_USE_OMP
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(! omp_in_parallel(), "RNG on GPU is not OpenMP thread safe");
#endif
        return rand_engine_descr;
    }

    struct RandomEngine {
        sycl_rng_engine* engine;
        AMREX_GPU_HOST_DEVICE RandomEngine (sycl_rng_engine* e = nullptr) : engine(e) {}
    };
#endif

#else

#ifdef AMREX_USE_HIP
    using randState_t = hiprandState_t;
    using randGenerator_t = hiprandGenerator_t;
#else
    using randState_t = curandState_t;
    using randGenerator_t = curandGenerator_t;
#endif

    extern randState_t* gpu_rand_state;

    inline
    randState_t* getRandState () {
#ifdef AMREX_USE_OMP
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(! omp_in_parallel(), "RNG on GPU is not OpenMP thread safe");
#endif
        return gpu_rand_state;
    }

    struct RandomEngine {
        randState_t* rand_state;
        AMREX_GPU_HOST_DEVICE RandomEngine (randState_t* rs) : rand_state(rs) {}
    };

#endif

#else

    struct RandomEngine {}; // CPU

#endif

    AMREX_FORCE_INLINE
    RandomEngine getInvalidRandomEngine () {
#ifdef AMREX_USE_GPU
        return RandomEngine{nullptr};
#else
        return RandomEngine{};
#endif
    }

}

#endif
