# WarpX-on-Metal Benchmark Results

**Hardware:** Apple M4 Pro (12 CPU cores, 16 GPU CUs, 24 GB unified memory)
**Date:** 2026-03-07 20:54:41

## GPU vs CPU Comparison

CPU baseline uses 12 OpenMP threads (Apple Clang, libomp).
GPU uses Metal via AdaptiveCpp SSCP. GPU times include first-run JIT warm-up cost;
timed runs use cached JIT. All results are median of 3 runs.

| Test | Grid | Particles/cell | Steps | CPU (12T) s/step | GPU (Metal) s/step | Speedup |
|------|------|---------------|-------|-------------------|---------------------|---------|
| langmuir_2d_small | 128x128 | 2^2=2*2 | 40 | 0.0027 | 0.0172 | 0.16x |
| langmuir_2d_large | 512x512 | 2^2=2*2 | 40 | 0.0173 | 0.0208 | 0.83x |
| langmuir_3d_small | 64^3 | 1^3 | 20 | 0.0081 | 0.0181 | 0.45x |
| langmuir_3d_large | 128^3 | 1^3 | 20 | 0.0560 | 0.0402 | 1.39x |

## Kernel Breakdown (GPU)

Source: `gpu_run_2`

```
  Level 0   1 grids  2097152 cells  100 % of domain
TinyProfiler total time across processes [min...avg...max]: 1.336 ... 1.336 ... 1.336
main()                                                          1     0.2822     0.2822     0.2822  21.13%
VisMF::Write(FabArray)                                          2     0.1535     0.1535     0.1535  11.50%
ParticleContainer::WriteParticles()                             4     0.1385     0.1385     0.1385  10.37%
FillBoundary_nowait()                                         320     0.1148     0.1148     0.1148   8.59%
FabArray::setVal()                                             87    0.08687    0.08687    0.08687   6.50%
PhysicalParticleContainer::Evolve::GatherAndPush               40    0.07239    0.07239    0.07239   5.42%
WarpX::EvolveB()                                               40    0.06765    0.06765    0.06765   5.07%
WarpXParticleContainer::DepositCurrent::CurrentDeposition      40    0.05685    0.05685    0.05685   4.26%
PhysicalParticleContainer::AddPlasma()                          2    0.05628    0.05628    0.05628   4.21%
ParticleCopyPlan::build                                        42    0.05158    0.05158    0.05158   3.86%
Redistribute_partition                                         42    0.04328    0.04328    0.04328   3.24%
WarpX::EvolveE()                                               20    0.03723    0.03723    0.03723   2.79%
ParticleContainer::SortParticlesByBin()                        10    0.03365    0.03365    0.03365   2.52%
DenseBins<T>::buildGPU                                         12    0.02394    0.02394    0.02394   1.79%
WriteBinaryParticleData()                                       4    0.02015    0.02015    0.02015   1.51%
sample::Coarsen()                                              22    0.01823    0.01823    0.01823   1.36%
PhysicalParticleContainer::PushP()                              4    0.01609    0.01609    0.01609   1.21%
ParticleContainer::addParticles                                 4    0.01477    0.01477    0.01477   1.11%
ParticleContainer::RedistributeGPU()                           42    0.01041    0.01041    0.01041   0.78%
FlushFormatPlotfile::WriteToFile()                              2   0.009714   0.009714   0.009714   0.73%
amrex::packBuffer                                              42   0.007275   0.007275   0.007275   0.54%
amrex::Add()                                                    4    0.00331    0.00331    0.00331   0.25%
Diagnostics::FilterComputePackFlush()                          22    0.00303    0.00303    0.00303   0.23%
amrex::ParticleToMesh                                           4   0.002409   0.002409   0.002409   0.18%
WarpX::InitData()                                               1   0.001111   0.001111   0.001111   0.08%
PhysicalParticleContainer::Evolve()                            40  0.0006021  0.0006021  0.0006021   0.05%
WarpX::OneStep_nosub()                                         20  0.0003957  0.0003957  0.0003957   0.03%
WriteMultiLevelPlotfile()                                       2  0.0003737  0.0003737  0.0003737   0.03%
WarpX::Evolve::step                                            20  0.0003636  0.0003636  0.0003636   0.03%
ablastr::utils::communication::FillBoundary                   258  0.0003229  0.0003229  0.0003229   0.02%
WarpX::SyncCurrent()                                           20  0.0001106  0.0001106  0.0001106   0.01%
FabArray::FillBoundaryAndSync()                               120  7.941e-05  7.941e-05  7.941e-05   0.01%
FillBoundaryAndSync_nowait()                                  120  6.259e-05  6.259e-05  6.259e-05   0.00%
FabArray::FillBoundary()                                      138  5.659e-05  5.659e-05  5.659e-05   0.00%
FabArray<FAB>::SumBoundary()                                   62  4.342e-05  4.342e-05  4.342e-05   0.00%
ablastr::utils::communication::SumBoundary                     62  3.963e-05  3.963e-05  3.963e-05   0.00%
WarpX::Evolve()                                                 1  3.663e-05  3.663e-05  3.663e-05   0.00%
FabArray<FAB>::SumBoundary_nowait()                            62  3.212e-05  3.212e-05  3.212e-05   0.00%
```

## Sort Interval Sensitivity

Run `./scripts/08-benchmark.sh` with sort_intervals overrides to populate:

| sort_intervals | Test | GPU s/step |
|---------------|------|-----------|
| 4 (default) | langmuir_2d_small | (run to populate) |
| -1 (disabled) | langmuir_2d_small | (run to populate) |
| 20 | langmuir_2d_small | (run to populate) |

## Notes

- GPU first-run includes JIT compilation (LLVM IR -> MSL). Subsequent runs use cache at `~/.acpp/apps/global/jit-cache/`.
- Single precision (FP32) throughout — Metal has no FP64 support.
- PSATD spectral solver unavailable (no Metal FFT). FDTD only.
- CPU build: Apple Clang, libomp, `-O3 -DNDEBUG`.
- GPU build: AdaptiveCpp SSCP + LLVM 20 -> MSL -> Apple GPU.
