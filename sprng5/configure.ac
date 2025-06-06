#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_INIT(sprng,5)
AC_CONFIG_SRCDIR(SRC/lfg/lfg.cpp)
AC_CANONICAL_SYSTEM

AC_CONFIG_MACRO_DIRS([m4])

AM_INIT_AUTOMAKE()
AM_CONFIG_HEADER(config.h:config.in)

#to make checks work when building outside source directory
AC_CONFIG_LINKS([check/cmrg/cmrg.data:check/cmrg/cmrg.data])
AC_CONFIG_LINKS([check/cmrg/F77/cmrg.data:check/cmrg/F77/cmrg.data])
AC_CONFIG_LINKS([check/lcg/lcg.data:check/lcg/lcg.data])
AC_CONFIG_LINKS([check/lcg/F77/lcg.data:check/lcg/F77/lcg.data])
AC_CONFIG_LINKS([check/lcg64/lcg64.data:check/lcg64/lcg64.data])
AC_CONFIG_LINKS([check/lcg64/F77/lcg64.data:check/lcg64/F77/lcg64.data])
AC_CONFIG_LINKS([check/lfg/lfg.data:check/lfg/lfg.data])
AC_CONFIG_LINKS([check/lfg/F77/lfg.data:check/lfg/F77/lfg.data])
AC_CONFIG_LINKS([check/mlfg/mlfg.data:check/mlfg/mlfg.data])
AC_CONFIG_LINKS([check/mlfg/F77/mlfg.data:check/mlfg/F77/mlfg.data])
AC_CONFIG_LINKS([check/pmlcg/pmlcg.data:check/pmlcg/pmlcg.data])
AC_CONFIG_LINKS([check/pmlcg/F77/pmlcg.data:check/pmlcg/F77/pmlcg.data])

AC_ARG_WITH(fortran, [AC_HELP_STRING([--with-fortran],[enable fortran compilation])],
                  with_fortran=$withval, with_fortran=no)  

AC_ARG_WITH(mpi, [AC_HELP_STRING([--with-mpi],[enable MPI parallelization])],
                  with_mpi=$withval, with_mpi=no)

if test "x$with_mpi" = "xyes"; then  
  use_mpi=y
else
  use_mpi=n 
fi

  # Checks for programs.
  AX_PROG_CXX_MPI([test x"$with_mpi" != xno], [use_mpi=y],[
  use_mpi=n
  if test x"$with_mpi" = xyes; then
    AC_MSG_FAILURE([MPI compiler requested, but couldn't use MPI.])
  else
    AC_MSG_WARN([Not using MPI.])
  fi
  ])
  AC_PROG_CC
  AC_PROG_CPP
  if test "$with_fortran" = "yes"; then  
    AX_PROG_F77_MPI([test x"$with_mpi" != xno], [use_mpi=y],[
    use_mpi=n
    if test x"$with_mpi" = xyes; then
      AC_MSG_FAILURE([MPI compiler requested, but couldn't use MPI.])
    else
      AC_MSG_WARN([Not using MPI.])
    fi
    ])
  fi




AC_F77_LIBRARY_LDFLAGS 

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lm':
AC_CHECK_LIB([m], [main])

AC_PROG_RANLIB
AC_PATH_PROGS(BASH, bash)

AC_SUBST(sprng_cxxflags)
AC_SUBST(sprng_fflags)

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([limits.h stddef.h stdlib.h string.h sys/time.h])

AC_CHECK_SIZEOF([long int])

CXXFLAGS="$CXXFLAGS -Wno-deprecated"

if test "$ac_cv_sizeof_long_int" = "8"; then
  CXXFLAGS="$CXXFLAGS -DLONG64=long"
fi
			  
AM_CONDITIONAL([USE_MPI], [test $use_mpi = y])

if test $use_mpi = y; then
  MPI_DEF="-DSPRNG_MPI"
  MPI_CXXLIB="-lmpi_cxx"
else
  MPI_DEF=""
  MPI_CXXLIB=""
fi

AC_SUBST(MPI_DEF)
AC_SUBST(MPI_CXXLIB)

AC_ARG_WITH(testu01, [AC_HELP_STRING([--with-testu01], [path to testU01])], 
                  with_testu01=$withval, with_testu01=no)
if test "$with_testu01" = "no"; then
  use_testu01=n
else
  use_testu01=y
  TESTU01_LIBDIR=$with_testu01"/lib"
  TESTU01_INCDIR=$with_testu01"/include"
fi                 

AC_ARG_WITH(testu01lib, [AC_HELP_STRING([--with-testu01lib], [path to testU01 libraries])], 
                  with_testu01lib=$withval, with_testu01lib=no)
AC_ARG_WITH(testu01inc, [AC_HELP_STRING([--with-testu01inc], [path to testU01 include directory])], 
                  with_testu01inc=$withval, with_testu01inc=no)
if test "$with_testu01lib" != "no"; then
  TESTU01_LIBDIR=$with_testu01lib
  if test "$with_testu01inc" != "no"; then
    use_testu01=y
    TESTU01_INCDIR=$with_testu01inc
  fi                 
fi

AC_SUBST(TESTU01_LIBDIR)
AC_SUBST(TESTU01_INCDIR)

AM_CONDITIONAL(USE_TESTU01, test "$use_testu01" = "y")

                
                  
if test "$with_fortran" = "yes"; then  
  if test -z "$F77"; then
    use_fortran=y
  fi
else
  use_fortran=n 
fi


AM_CONDITIONAL(compile_FORTRAN, test "$use_fortran" = "y")

AC_CHECK_HEADER(fft.h,fft="yes",fft="no")
if test $fft="yes"; then
  AC_CHECK_FUNC(dzfft2dui, [fft="yes"], [fft="no"])
fi
AM_CONDITIONAL(FFT, test $fft = "yes")

AC_C_BIGENDIAN

dnl deal with Fortran pointer size
AC_CHECK_SIZEOF(int *)
if test "$ac_cv_sizeof_int_p" = "8"; then
  FFPSIZE="-DPOINTER_SIZE=8 -DLONG64=long -DINTEGER_STAR_8"
else
  FFPSIZE=""  
fi

FMANGLE=""

AC_SUBST(FFPSIZE)
AC_SUBST(FMANGLE)

AC_CHECK_SIZEOF(double)
AC_CHECK_SIZEOF(float)

if test "$with_fortran" != "yes"; then  
  F77="echo"
  FORTRANDIR=
else
  FORTRANDIR="F77"

AC_F77_WRAPPERS dnl provides F77_FUNC and  F77_FUNC_ for name mangling
fi

AC_SUBST(FORTRANDIR)

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T
AC_STRUCT_TM

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_CHECK_FUNCS([memset pow sqrt])

AC_PROG_INSTALL
AC_OUTPUT([Makefile 
	     include/Makefile
	     lib/Makefile
	     SRC/Makefile
	     SRC/cmrg/Makefile
	     SRC/lcg/Makefile
	     SRC/lcg64/Makefile
	     SRC/lfg/Makefile
	     SRC/mlfg/Makefile
	     SRC/pmlcg/Makefile
		 check/Makefile 
		 check/lcg/Makefile 
		 check/lcg/F77/Makefile 
		 check/lfg/Makefile
		 check/lfg/F77/Makefile
		 check/mlfg/Makefile
		 check/mlfg/F77/Makefile
		 check/cmrg/Makefile
		 check/cmrg/F77/Makefile
		 check/lcg64/Makefile
		 check/lcg64/F77/Makefile
		 check/pmlcg/Makefile
		 check/pmlcg/F77/Makefile
		 TESTS/Makefile
		 TESTS/mpitests/Makefile
     TESTS/testu01/Makefile
     EXAMPLES/Makefile
		 EXAMPLES/mpisprng/Makefile
		 EXAMPLES/F77/Makefile])

