# Apple-Silicon (Metal) GPU Port of WarpX

## Purpose

This document explains the portability shape of the Apple-Silicon GPU port of WarpX: AMReX SYCL -> AdaptiveCpp SSCP -> Metal.

Portability contract: every change to a shared AMReX or WarpX code path is behind an Apple/Metal `#if` gate. The non-Metal branch is the exact upstream code. Linux CUDA, HIP, CPU, and oneAPI SYCL builds therefore compile and behave identically to vanilla AMReX/WarpX for these shared paths.

Verification status:

- Non-Metal branches were diffed against vanilla upstream; residuals: none.
- Metal physics is unchanged: Langmuir GPU results match CPU at about `1e-6` for both 1 and 4 particles/cell.
- The CPU build compiles and runs a finite tiny case.

## Apple/Metal Gate Macro

The shared gate is `AMREX_METAL_PORT`, defined in `Tools/CMake/AMReX_Config.H.in`:

```diff
+#if defined(AMREX_USE_SYCL) && defined(__APPLE__) && \
+    (defined(__ADAPTIVECPP__) || defined(__ACPP__) || \
+     defined(SYCL_IMPLEMENTATION_ACPP) || defined(SYCL_IMPLEMENTATION_HIPSYCL) || \
+     defined(__HIPSYCL__))
+#  define AMREX_METAL_PORT 1
+#endif
```

This is true only for Apple builds using the SYCL path with AdaptiveCpp-compatible compiler macros. It is false for CUDA, HIP, CPU-only, and oneAPI SYCL builds.

## Gated Changes

### 2. DenseBins Storage

Metal path: `DenseBinIteratorFactory` gains device-safe pointer state and constructors, and `DenseBins` stores bin arrays in `Gpu::ManagedVector` instead of `Gpu::DeviceVector`.

Why needed: AdaptiveCpp SSCP's Metal lowering needs the nested pointer state used by bin iterators to come from memory the backend can translate and access consistently. The managed/buffer-backed storage and explicit raw-pointer factory state make the iterator state visible to Metal kernels without changing other backends.

Files and symbols: `Src/Particle/AMReX_DenseBins.H`, `DenseBinIteratorFactory`, `DenseBins`.

```diff
+#ifdef AMREX_METAL_PORT
+    AMREX_GPU_HOST_DEVICE
+    DenseBinIteratorFactory () = default;
+
+    template <typename OffsetVector, typename PermutationVector>
+    DenseBinIteratorFactory (const OffsetVector& offsets,
+                             const PermutationVector& permutation,
+                             const T* items)
+        : m_offsets_ptr(offsets.dataPtr()),
+          m_permutation_ptr(permutation.dataPtr()),
+          m_items(items),
+          m_num_items(static_cast<int>(permutation.size()))
+    {}
+
+    AMREX_GPU_HOST_DEVICE
+    DenseBinIteratorFactory (const index_type* offsets,
+                             const index_type* permutation,
+                             const_pointer_type items,
+                             int num_items) noexcept
+        : m_offsets_ptr(offsets),
+          m_permutation_ptr(permutation),
+          m_items(items),
+          m_num_items(num_items)
+    {}
+#else
     DenseBinIteratorFactory (const Gpu::DeviceVector<index_type>& offsets,
                              const Gpu::DeviceVector<index_type>& permutation,
                              const T* items)
@@ -44,6 +69,7 @@ struct DenseBinIteratorFactory
           m_permutation_ptr(permutation.dataPtr()),
           m_items(items)
     {}
+#endif
```

```diff
+#ifdef AMREX_METAL_PORT
+    const index_type* m_offsets_ptr = nullptr;
+    const index_type* m_permutation_ptr = nullptr;
+    const_pointer_type m_items = nullptr;
+    int m_num_items = 0;
+#else
     const index_type* m_offsets_ptr;
     const index_type* m_permutation_ptr;
     const_pointer_type m_items;
+#endif
```

```diff
+#ifdef AMREX_METAL_PORT
+    Gpu::ManagedVector<index_type> m_bins;
+    Gpu::ManagedVector<index_type> m_counts;
+    Gpu::ManagedVector<index_type> m_local_offsets;
+    Gpu::ManagedVector<index_type> m_offsets;
+    Gpu::ManagedVector<index_type> m_perm;
+#else
     Gpu::DeviceVector<index_type> m_bins;
     Gpu::DeviceVector<index_type> m_counts;
     Gpu::DeviceVector<index_type> m_local_offsets;
     Gpu::DeviceVector<index_type> m_offsets;
     Gpu::DeviceVector<index_type> m_perm;
+#endif
```

The `#else` branch is the upstream `Gpu::DeviceVector` storage and original factory state.

### 3. AMReX No-Filter Particle Copy

Metal path: the no-filter overloads of `copyParticles` and `addParticles` avoid the predicate-copy path and do a direct tile allocation/copy, followed by synchronization and optional redistribution.

Why needed: the Metal random-engine predicate-copy path under-copied particles when no filter was actually requested. The direct no-filter path avoids invoking a predicate and random engine for the all-particles case.

Files and symbols: `Src/Particle/AMReX_ParticleContainerI.H`, `ParticleContainer_impl::copyParticles`, `ParticleContainer_impl::addParticles`.

```diff
@@ -1143,8 +1143,14 @@ void
 ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>::
 copyParticles (const PCType& other, bool local)
 {
+#ifdef AMREX_METAL_PORT
+    BL_PROFILE("ParticleContainer::copyParticles");
+    clearParticles();
+    addParticles(other, local);
+#else
     using PData = typename ParticleTileType::ConstParticleTileDataType;
     copyParticles(other, [] AMREX_GPU_HOST_DEVICE (const PData& /*data*/, int /*i*/) { return 1; }, local);
+#endif
 }
```

```diff
@@ -1154,8 +1160,49 @@ void
 ParticleContainer_impl<ParticleType, NArrayReal, NArrayInt, Allocator, CellAssignor>::
 addParticles (const PCType& other, bool local)
 {
+#ifdef AMREX_METAL_PORT
+    BL_PROFILE("ParticleContainer::addParticles");
+
+    for (int lev = 0; lev < other.numLevels(); ++lev)
+    {
+        [[maybe_unused]] Gpu::NoSyncRegion no_sync{};
+        const auto& plevel_other = other.GetParticles(lev);
+        for (MFIter mfi = other.MakeMFIter(lev); mfi.isValid(); ++mfi)
+        {
+            auto index = std::make_pair(mfi.index(), mfi.LocalTileIndex());
+            if (!plevel_other.contains(index)) { continue; }
+            DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
+        }
+    }
+
+#ifdef AMREX_USE_OMP
+#pragma omp parallel if (Gpu::notInLaunchRegion())
+#endif
+    for (int lev = 0; lev < other.numLevels(); ++lev)
+    {
+        const auto& plevel_other = other.GetParticles(lev);
+        for (MFIter mfi = other.MakeMFIter(lev); mfi.isValid(); ++mfi)
+        {
+            auto index = std::make_pair(mfi.index(), mfi.LocalTileIndex());
+            if (!plevel_other.contains(index)) { continue; }
+
+            auto& ptile = ParticlesAt(lev, mfi.index(), mfi.LocalTileIndex());
+            const auto& ptile_other = plevel_other.at(index);
+            auto np = ptile_other.numParticles();
+            if (np == 0) { continue; }
+
+            auto dst_index = ptile.numParticles();
+            ptile.resize(dst_index + np);
+            amrex::copyParticles(ptile, ptile_other, 0, dst_index, np);
+        }
+    }
+
+    Gpu::streamSynchronize();
+    if (! local) { Redistribute(); }
+#else
     using PData = typename ParticleTileType::ConstParticleTileDataType;
     addParticles(other, [] AMREX_GPU_HOST_DEVICE (const PData& /*data*/, int /*i*/) { return 1; }, local);
+#endif
 }
```

The `#else` branch is the upstream predicate-delegating implementation.

### 4. ParticleLocator Managed/Device Construction

Metal path: locator backing storage uses managed vectors where Metal kernels need to dereference locator state, and grid assignors are constructed on the device from explicit pointer fields.

Why needed: the Metal SSCP path is sensitive to nested pointer translation and host-constructed objects that contain device pointers. Constructing `AssignGrid` state on the device, with managed/buffer-backed storage for the containing vectors, avoids leaking host-side pointer assumptions into Metal kernels.

Files and symbols: `Src/Particle/AMReX_ParticleLocator.H`, `AssignGrid`, `ParticleLocator`, `ParticleLocatorHierarchy`.

```diff
+#ifdef AMREX_METAL_PORT
+    AMREX_GPU_HOST_DEVICE
+    AssignGrid (BinIteratorFactory a_bif, Dim3 a_bins_lo, Dim3 a_bins_hi,
+                Dim3 a_bin_size, Dim3 a_num_bins, Box const& a_domain,
+                GpuArray<Real, AMREX_SPACEDIM> const& a_plo,
+                GpuArray<Real, AMREX_SPACEDIM> const& a_dxi,
+                bool a_do_tiling, Dim3 a_tile_size) noexcept
+        : m_bif(a_bif),
+          m_lo(a_bins_lo), m_hi(a_bins_hi), m_bin_size(a_bin_size),
+          m_num_bins(a_num_bins), m_domain(a_domain),
+          m_plo(a_plo), m_dxi(a_dxi),
+          m_tile_size(a_tile_size), m_do_tiling(a_do_tiling)
+        {
+            // clamp bin size and num_bins to 1 for AMREX_SPACEDIM < 3
+            if (m_bin_size.x >= 0) {m_bin_size.x = amrex::max(m_bin_size.x, 1);}
+            if (m_bin_size.y >= 0) {m_bin_size.y = amrex::max(m_bin_size.y, 1);}
+            if (m_bin_size.z >= 0) {m_bin_size.z = amrex::max(m_bin_size.z, 1);}
+
+            if (m_bin_size.x >= 0) {m_num_bins.x = amrex::max(m_num_bins.x, 1);}
+            if (m_bin_size.y >= 0) {m_num_bins.y = amrex::max(m_num_bins.y, 1);}
+            if (m_bin_size.z >= 0) {m_num_bins.z = amrex::max(m_num_bins.z, 1);}
+        }
+#endif
```

```diff
+#ifdef AMREX_METAL_PORT
+    Gpu::ManagedVector<Box> m_device_boxes;
+#else
     Gpu::DeviceVector<Box> m_device_boxes;
+#endif
```

```diff
+#ifdef AMREX_METAL_PORT
+    Gpu::ManagedVector<AssignGrid<BinIteratorFactory> > m_grid_assignors;
+#else
     Gpu::DeviceVector<AssignGrid<BinIteratorFactory> > m_grid_assignors;
+#endif
```

```diff
@@ -316,7 +403,14 @@ public:
         int num_levels = static_cast<int>(a_ba.size());
         m_locators.resize(num_levels);
         m_grid_assignors.resize(num_levels);
-#ifdef AMREX_USE_GPU
+#ifdef AMREX_METAL_PORT
+        for (int lev = 0; lev < num_levels; ++lev)
+        {
+            m_locators[lev].build(a_ba[lev], a_geom[lev], a_do_tiling, a_tile_size);
+            m_locators[lev].constructGridAssignorOnDevice(m_grid_assignors.dataPtr() + lev);
+        }
+        Gpu::streamSynchronize();
+#elif defined(AMREX_USE_GPU)
         Gpu::HostVector<AssignGrid<BinIteratorFactory> > h_grid_assignors(num_levels);
         for (int lev = 0; lev < num_levels; ++lev)
         {
```

```diff
@@ -379,7 +473,14 @@ public:
     void setGeometry (const ParGDBBase* a_gdb)
     {
         int num_levels = a_gdb->finestLevel()+1;
-#ifdef AMREX_USE_GPU
+#ifdef AMREX_METAL_PORT
+        for (int lev = 0; lev < num_levels; ++lev)
+        {
+            m_locators[lev].setGeometry(a_gdb->Geom(lev));
+            m_locators[lev].constructGridAssignorOnDevice(m_grid_assignors.dataPtr() + lev);
+        }
+        Gpu::streamSynchronize();
+#elif defined(AMREX_USE_GPU)
         Gpu::HostVector<AssignGrid<BinIteratorFactory> > h_grid_assignors(num_levels);
         for (int lev = 0; lev < num_levels; ++lev)
         {
```

The `#else` and `#elif defined(AMREX_USE_GPU)` branches keep the upstream `Gpu::DeviceVector`, host `h_grid_assignors`, and host-to-device copy construction path for non-Metal builds.

### 6. WarpX Diagnostic No-Filter Fast Path

Metal path: particle diagnostics check whether any particle filter is active. If no filter is active, diagnostics use the no-filter copy overload instead of invoking the random-engine predicate path.

Why needed: the Metal random-engine predicate-copy path under-copied particles for the no-filter diagnostic case. The fast path avoids the predicate and random engine when the diagnostic is semantically "copy all particles."

Files and symbols: `Source/Diagnostics/FlushFormats/FlushFormatPlotfile.cpp`, `FlushFormatPlotfile::WriteParticles`.

```diff
@@ -438,6 +438,24 @@ FlushFormatPlotfile::WriteParticles(const std::string& dir,
         if (!isBTD) {
             particlesConvertUnits(ConvertDirection::WarpX_to_SI, pc, mass);
             using SrcData = WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
+#ifdef AMREX_METAL_PORT
+            const bool has_particle_filter = part_diag.m_do_random_filter ||
+                part_diag.m_do_uniform_filter || part_diag.m_do_parser_filter ||
+                part_diag.m_do_geom_filter;
+            if (has_particle_filter) {
+                tmp.copyParticles(*pc,
+                                  [random_filter,uniform_filter,parser_filter,geometry_filter]
+                                  AMREX_GPU_HOST_DEVICE
+                                  (const SrcData& src, int ip, const amrex::RandomEngine& engine)
+                {
+                    const SuperParticleType& p = src.getSuperParticle(ip);
+                    return random_filter(p, engine) * uniform_filter(p, engine)
+                        * parser_filter(p, engine) * geometry_filter(p, engine);
+                }, true);
+            } else {
+                tmp.copyParticles(*pc, true);
+            }
+#else
             tmp.copyParticles(*pc,
                               [random_filter,uniform_filter,parser_filter,geometry_filter]
                               AMREX_GPU_HOST_DEVICE
@@ -447,10 +465,29 @@ FlushFormatPlotfile::WriteParticles(const std::string& dir,
                 return random_filter(p, engine) * uniform_filter(p, engine)
                     * parser_filter(p, engine) * geometry_filter(p, engine);
             }, true);
+#endif
             particlesConvertUnits(ConvertDirection::SI_to_WarpX, pc, mass);
         } else {
             particlesConvertUnits(ConvertDirection::WarpX_to_SI, pinned_pc, mass);
             using SrcData = WarpXParticleContainer::ParticleTileType::ConstParticleTileDataType;
+#ifdef AMREX_METAL_PORT
+            const bool has_particle_filter = part_diag.m_do_random_filter ||
+                part_diag.m_do_uniform_filter || part_diag.m_do_parser_filter ||
+                part_diag.m_do_geom_filter;
+            if (has_particle_filter) {
+                tmp.copyParticles(*pinned_pc,
+                                  [random_filter,uniform_filter,parser_filter,geometry_filter]
+                                  AMREX_GPU_HOST_DEVICE
+                                  (const SrcData& src, int ip, const amrex::RandomEngine& engine)
+                {
+                    const SuperParticleType& p = src.getSuperParticle(ip);
+                    return random_filter(p, engine) * uniform_filter(p, engine)
+                        * parser_filter(p, engine) * geometry_filter(p, engine);
+                }, true);
+            } else {
+                tmp.copyParticles(*pinned_pc, true);
+            }
+#else
             tmp.copyParticles(*pinned_pc,
                               [random_filter,uniform_filter,parser_filter,geometry_filter]
                               AMREX_GPU_HOST_DEVICE
@@ -460,6 +497,7 @@ FlushFormatPlotfile::WriteParticles(const std::string& dir,
                 return random_filter(p, engine) * uniform_filter(p, engine)
                     * parser_filter(p, engine) * geometry_filter(p, engine);
             }, true);
+#endif
             particlesConvertUnits(ConvertDirection::SI_to_WarpX, pinned_pc, mass);
         }
```

The `#else` branch is the upstream predicate-copy diagnostic path for both `pc` and `pinned_pc`.

## Upstream-Benefiting Reverts

These changes are applied unconditionally because they are backend-neutral cleanups or restorations, not Metal-specific workarounds.

- `Random_int` / `Random_long` `n == 0` guard: restores the upstream host guard that returns 0 before `std::uniform_int_distribution(0, n - 1)` can underflow. This fixes a latent host-side correctness issue for all backends.
- `partitionParticlesByDest` unused argument removal: removes an unused `dx` argument and restores the API/callsite shape to the upstream form. This is an API cleanup, not an Apple-specific behavior change.
- `TotalNumberOfParticles(false, true)` debug probes: removes discarded particle-count calls whose results were unused. This avoids dead synchronization/counting work for every backend.

## Coarse-Gated Changes

These changes already exclude CUDA, HIP, and CPU builds. They are gated to SYCL plus AdaptiveCpp-compatible paths rather than strictly to Apple. If strictly Apple-only behavior is desired, these gates can be tightened with `__APPLE__`.

- Parser single-precision executor: `ParserExeReal=float` is selected under `AMREX_USE_SYCL && AMREX_USE_FLOAT && AdaptiveCpp-compatible SYCL`; non-SYCL builds keep the upstream `double` executor path.
- fp32 atomic-add deposition: AdaptiveCpp-compatible SYCL atomic add paths are enabled in `Src/Base/AMReX_GpuAtomic.H`; CUDA/HIP/CPU atomic paths are not selected by this gate.
- Custom Philox-like SYCL RNG engine: `acpp_rng_engine` and related RNG dispatch are selected for AdaptiveCpp-compatible SYCL where oneAPI MKL RNG is not the active backend.
- `sort_intervals` default disable: WarpX disables the default particle sort interval only under `AMREX_USE_GPU && AMREX_USE_SYCL && AdaptiveCpp-compatible SYCL`; CUDA/HIP/CPU defaults remain upstream.
- Host-side parser-momentum repair: `Source/Particles/ParticleCreation/AddParticles.cpp` repairs XZ parser-momentum handling under `AMREX_USE_GPU && AMREX_USE_SYCL && WARPX_DIM_XZ`; CUDA/HIP/CPU builds do not enter this path.
- int128 opt-out: `AMREX_NO_INT128` is defined only in the AdaptiveCpp SYCL configuration path; other backends keep the default integer support path.

## Portability Argument

For the four shared-path accommodations above, the branch structure is:

```cpp
#ifdef AMREX_METAL_PORT
    // Apple/Metal accommodation
#else
    // vanilla upstream code
#endif
```

or the equivalent:

```cpp
#ifdef AMREX_METAL_PORT
    // Apple/Metal accommodation
#elif defined(AMREX_USE_GPU)
    // vanilla upstream GPU code
#endif
```

Because `AMREX_METAL_PORT` is false for CUDA, HIP, CPU-only, and oneAPI builds, those builds take the same code that vanilla upstream takes. CUDA is not buildable on Apple hardware, so CUDA portability is covered by construction through the upstream-equivalent non-Metal branch. The CPU build is additionally covered by a compile-and-run check.

## Upstream Roadmap

1. AdaptiveCpp: land the SSCP-to-Metal compiler/runtime backend contract first. This is the foundation for the AMReX and WarpX accommodations.
2. AMReX: upstream small, tested SYCL/Metal support PRs for RNG, parser precision, atomics, and particle infrastructure, while preserving non-Metal CI behavior.
3. WarpX: upstream only the small accommodations that remain necessary after the backend and AMReX layers settle, with tests and Apple/Metal gates.

The system-memory-guard change is already submitted upstream and should stay separate from the particle and parser accommodations.
