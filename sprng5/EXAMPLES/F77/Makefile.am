.NOTPARALLEL:

LIBDIR=$(top_builddir)/lib
SRCDIR=$(top_srcdir)/SRC
INCLUDEDIR=$(top_srcdir)/include
INCLUDE = -I$(INCLUDEDIR)

GEN=sprng
CLEANFILES=*.$(GEN)

bin_PROGRAMS = convertf.tmp pif-simple.tmp seedf.tmp seedf-simple.tmp simplef-simple.tmp spawnf.tmp sprngf.tmp sprngf-simple.tmp subroutinef.tmp


convertf_tmp_SOURCES = convertf.F ../../include/sprng_f.h genf_types_menu.h
convertf_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR)
convertf_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
convertf_tmp_LDFLAGS = 

pif_simple_tmp_SOURCES = pif-simple.F ../../include/sprng_f.h genf_types_menu.h
pif_simple_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR)
pif_simple_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
pif_simple_tmp_LDFLAGS = 

seedf_tmp_SOURCES = seedf.F ../../include/sprng_f.h genf_types_menu.h
seedf_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR)
seedf_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
seedf_tmp_LDFLAGS = 

seedf_simple_tmp_SOURCES = seedf-simple.F ../../include/sprng_f.h genf_types_menu.h
seedf_simple_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
seedf_simple_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
seedf_simple_tmp_LDFLAGS = 

simplef_simple_tmp_SOURCES = simplef-simple.F ../../include/sprng_f.h genf_types_menu.h
simplef_simple_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
simplef_simple_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
simplef_simple_tmp_LDFLAGS = 

spawnf_tmp_SOURCES = spawnf.F ../../include/sprng_f.h genf_types_menu.h
spawnf_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
spawnf_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
spawnf_tmp_LDFLAGS = 

subroutinef_tmp_SOURCES = subroutinef.F ../../include/sprng_f.h genf_types_menu.h
subroutinef_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
subroutinef_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
subroutinef_tmp_LDFLAGS = 

sprngf_tmp_SOURCES = sprngf.F ../../include/sprng_f.h genf_types_menu.h
sprngf_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
sprngf_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
sprngf_tmp_LDFLAGS = 

sprngf_simple_tmp_SOURCES = sprngf-simple.F ../../include/sprng_f.h genf_types_menu.h
sprngf_simple_tmp_FFLAGS = $(FFPSIZE) -I$(INCLUDEDIR) 
sprngf_simple_tmp_LDADD = -L$(LIBDIR) -l$(GEN) -lstdc++ $(MPI_CXXLIB)
sprngf_simple_tmp_LDFLAGS = 


bin_SCRIPTS = sprngtest
sprngtest:
	@rm -f *.o
	@mv convertf.tmp convertf.$(GEN)
	@mv pif-simple.tmp pif-simple.$(GEN)
	@mv seedf.tmp seedf.$(GEN)
	@mv seedf-simple.tmp seedf-simple.$(GEN)
	@mv simplef-simple.tmp simplef-simple.$(GEN)
	@mv spawnf.tmp spawnf.$(GEN)
	@mv sprngf.tmp sprngf.$(GEN)
	@mv sprngf-simple.tmp sprngf-simple.$(GEN)
	@mv subroutinef.tmp subroutinef.$(GEN)
