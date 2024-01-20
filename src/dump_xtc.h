/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */


#ifndef PYXTC_DUMP_XTC_H
#define PYXTC_DUMP_XTC_H

#include <stdio.h>
#include "xdr_compat.h"


#define MAXSMALLINT INT_MAX

// namespace PYXTC_NS {

XDR* xdropen(char*, const char *);
int xdrclose(XDR *);
void xdrfreebuf();
bool_t xdr_header(XDR *,int *,int *,float *,float *);
int xdr3dfcoord(XDR *, float *, int *, float *);
unsigned int xdrgetpos(XDR *);
bool_t xdrsetpos(XDR*,unsigned int*);
// }

#endif
