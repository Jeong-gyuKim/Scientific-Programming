
SUBDIRS = .

.NOTPARALLEL:

EXTRA_DIST=pmlcg.data .insertlib
INCLUDEDIR = $(top_srcdir)/SRC

LIBDIR=$(top_builddir)/lib
CHKDIR=$(top_srcdir)/check

bin_PROGRAMS = check.fpmlcg check_sim.fpmlcg time.fpmlcg
check_fpmlcg_SOURCES = check_genf.F 
check_fpmlcg_DEPENDENCIES = $(LIBDIR)/libsprng.a
check_fpmlcg_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
check_fpmlcg_LDADD = -L$(LIBDIR) -lsprng -lstdc++ $(MPI_CXXLIB)
check_fpmlcg_LDFLAGS = 
check_sim_fpmlcg_SOURCES = check_genf_simple.F 
check_sim_fpmlcg_DEPENDENCIES = $(LIBDIR)/libsprng.a
check_sim_fpmlcg_LDADD = -L$(LIBDIR) -lsprng -lstdc++ $(MPI_CXXLIB)
check_sim_fpmlcg_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
check_sim_fpmlcg_LDFLAGS = 
time_fpmlcg_SOURCES = timingf.F
time_fpmlcg_DEPENDENCIES = $(LIBDIR)/libsprng.a
time_fpmlcg_LDADD = -L$(LIBDIR) -lsprng -lstdc++ $(MPI_CXXLIB)
time_fpmlcg_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
time_fpmlcg_LDFLAGS = 

bin_SCRIPTS = checkpmlcg timepmlcg
CLEANFILES = $(bin_SCRIPTS)

checkpmlcg: $(CHKDIR)/insertlib
	rm -f ./checkpmlcg
	@$(CHKDIR)/insertlib pmlcg ./checkpmlcg "Checking Default FORTRAN interface:" "check.fpmlcg < ./pmlcg.data 2> /dev/null" ./check.fpmlcg
	@$(CHKDIR)/insertlib pmlcg ./checkpmlcg "Checking Simple FORTRAN interface:" "check_sim.fpmlcg < ./pmlcg.data 2> /dev/null" ./check_sim.fpmlcg
	chmod u+x ./checkpmlcg

timepmlcg: $(CHKDIR)/insertlib
	rm -f ./timepmlcg
	@$(CHKDIR)/insertlib pmlcg ./timepmlcg "Timing FORTRAN interface" ./time.fpmlcg ./time.fpmlcg
	chmod u+x ./timepmlcg

