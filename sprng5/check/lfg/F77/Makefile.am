
SUBDIRS = .

EXTRA_DIST=lfg.data .insertlib
INCLUDEDIR = $(top_srcdir)/SRC

LIBDIR=$(top_builddir)/lib
CHKDIR=$(top_srcdir)/check

.NOTPARALLEL:

bin_PROGRAMS = check.flfg check_sim.flfg time.flfg
check_flfg_SOURCES = check_genf.F 
check_flfg_DEPENDENCIES = $(LIBDIR)/libsprng.a
check_flfg_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR)
check_flfg_LDADD = -L$(LIBDIR) -lsprng -lstdc++ $(MPI_CXXLIB)
check_flfg_LDFLAGS =  
check_sim_flfg_SOURCES = check_genf_simple.F 
check_sim_flfg_DEPENDENCIES = $(LIBDIR)/libsprng.a
check_sim_flfg_LDADD = -L$(LIBDIR) -lsprng -lstdc++ $(MPI_CXXLIB)
check_sim_flfg_FFLAGS =$(FFPSIZE) -I$(INCLUDEDIR)
check_sim_flfg_LDFLAGS =  
time_flfg_SOURCES = timingf.F
time_flfg_DEPENDENCIES = $(LIBDIR)/libsprng.a
time_flfg_LDADD = -L$(LIBDIR) -lsprng -lstdc++ $(MPI_CXXLIB)
time_flfg_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR)
time_flfg_LDFLAGS =  

bin_SCRIPTS = checklfg timelfg
CLEANFILES = $(bin_SCRIPTS)

checklfg:
	rm -f ./checklfg
	chmod u+x $(CHKDIR)/insertlib
	@$(CHKDIR)/insertlib lfg ./checklfg "Checking Default FORTRAN interface:" "check.flfg < ./lfg.data 2> /dev/null" ./check.flfg
	@$(CHKDIR)/insertlib lfg ./checklfg "Checking Simple FORTRAN interface:" "check_sim.flfg < ./lfg.data 2> /dev/null" ./check_sim.flfg
	chmod u+x ./checklfg

timelfg: $(CHKDIR)/insertlib
	rm -f ./timelfg
	@$(CHKDIR)/insertlib lfg ./timelfg "Timing FORTRAN interface" ./time.flfg ./time.flfg
	chmod u+x ./timelfg
