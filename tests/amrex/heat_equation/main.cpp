// HeatEquation test — local copy with float-safe literals for Metal GPU.
// Metal does not support double; all literals must be float (or Real).
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

#include "myfunc.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    // **********************************
    // SIMULATION PARAMETERS

    // number of cells on each side of the domain
    int n_cell;

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;

    // inputs parameters
    {
        ParmParse pp;
        pp.get("n_cell",n_cell);
        pp.get("max_grid_size",max_grid_size);
        nsteps = 10;
        pp.query("nsteps",nsteps);
        plot_int = -1;
        pp.query("plot_int",plot_int);
        pp.get("dt",dt);
    }

    // **********************************
    // SIMULATION SETUP

    BoxArray ba;
    Geometry geom;

    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));

    Box domain(dom_lo, dom_hi);

    ba.define(domain);
    ba.maxSize(max_grid_size);

    // Use Real literals to avoid double promotion on Metal
    RealBox real_box({AMREX_D_DECL( Real(0.0), Real(0.0), Real(0.0))},
                     {AMREX_D_DECL( Real(1.0), Real(1.0), Real(1.0))});

    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    int Nghost = 1;
    int Ncomp = 1;

    DistributionMapping dm(ba);

    MultiFab phi_old(ba, dm, Ncomp, Nghost);
    MultiFab phi_new(ba, dm, Ncomp, Nghost);

    Real time = Real(0.0);

    // **********************************
    // INITIALIZE DATA

    for (MFIter mfi(phi_old); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        const Array4<Real>& phiOld = phi_old.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real x = (i+Real(0.5)) * dx[0];
            Real y = (j+Real(0.5)) * dx[1];
#if (AMREX_SPACEDIM == 2)
            Real rsquared = ((x-Real(0.5))*(x-Real(0.5))+(y-Real(0.5))*(y-Real(0.5)))/Real(0.01);
#elif (AMREX_SPACEDIM == 3)
            Real z= (k+Real(0.5)) * dx[2];
            Real rsquared = ((x-Real(0.5))*(x-Real(0.5))+(y-Real(0.5))*(y-Real(0.5))+(z-Real(0.5))*(z-Real(0.5)))/Real(0.01);
#endif
            phiOld(i,j,k) = Real(1.0) + std::exp(-rsquared);
        });
    }

    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, phi_old, {"phi"}, geom, time, 0);
    }

    for (int step = 1; step <= nsteps; ++step)
    {
        phi_old.FillBoundary(geom.periodicity());

        for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            const Array4<Real>& phiOld = phi_old.array(mfi);
            const Array4<Real>& phiNew = phi_new.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                phiNew(i,j,k) = phiOld(i,j,k) + dt *
                    ( (phiOld(i+1,j,k) - Real(2.0)*phiOld(i,j,k) + phiOld(i-1,j,k)) / (dx[0]*dx[0])
                     +(phiOld(i,j+1,k) - Real(2.0)*phiOld(i,j,k) + phiOld(i,j-1,k)) / (dx[1]*dx[1])
#if (AMREX_SPACEDIM == 3)
                     +(phiOld(i,j,k+1) - Real(2.0)*phiOld(i,j,k) + phiOld(i,j,k-1)) / (dx[2]*dx[2])
#endif
                        );
            });
        }

        time = time + dt;

        MultiFab::Copy(phi_old, phi_new, 0, 0, 1, 0);

        amrex::Print() << "Advanced step " << step << "\n";

        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, phi_new, {"phi"}, geom, time, step);
        }
    }
}
