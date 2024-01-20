/*
  Code taken from LAMMPs, respective licence (GPL) applies.
*/

/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: 
                         Miguel Amaral (ISTA)
                          modifications to allow incremental index construction
                         Naveen Michaud-Agrawal (Johns Hopkins U)
                         open-source XDR routines from
                           Frans van Hoesel (https://www.rug.nl/staff/f.h.j.van.hoesel/)
                           are included in this file
                         Axel Kohlmeyer (Temple U)
                           port to platforms without XDR support
                           added support for unwrapped trajectories
                           support for groups
------------------------------------------------------------------------- */

#include "dump_xtc.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// using namespace PYXTC_NS;

#define EPS 1e-5
#define XTC_MAGIC 1995

#define MYMIN(a,b) ((a) < (b) ? (a) : (b))
#define MYMAX(a,b) ((a) > (b) ? (a) : (b))

// int xdropen(XDR *, FILE*, const char *);
// int xdrclose(XDR *);
// void xdrfreebuf();
// bool_t xdr_header(XDR *xd,int *n,int *ntimestep,float *time_value,float *box);
// int xdr3dfcoord(XDR *, float *, int *, float *);


// ----------------------------------------------------------------------
// C functions that create GROMOS-compatible XDR files
// open-source
// (c) 1995 Frans van Hoesel, hoesel@chem.rug.nl
// ----------------------------------------------------------------------

/*____________________________________________________________________________
 |
 | Below are the routines to be used by C programmers. Use the 'normal'
 | xdr routines to write integers, floats, etc (see man xdr)
 |
 | int xdropen(XDR *xdrs, const char *filename, const char *type)
 |        This will open the file with the given filename and the
 |        given mode. You should pass it an allocated XDR struct
 |        in xdrs, to be used in all other calls to xdr routines.
 |        Mode is 'w' to create, or update an file, and for all
 |        other values of mode the file is opened for reading.
 |        You need to call xdrclose to flush the output and close
 |        the file.
 |
 |        Note that you should not use xdrstdio_create, which
 |        comes with the standard xdr library.
 |
 | int xdrclose(XDR *xdrs)
 |        Flush the data to the file, and close the file;
 |        You should not use xdr_destroy (which comes standard
 |        with the xdr libraries).
 |
 | int xdr3dfcoord(XDR *xdrs, float *fp, int *size, float *precision)
 |        This is \fInot\fR a standard xdr routine. I named it this
 |        way, because it invites people to use the other xdr
 |        routines.
 |
 |        (c) 1995 Frans van Hoesel, hoesel@chem.rug.nl
*/

// #define MAXID 20
// static FILE *xdrfiles[MAXID];
// static XDR *xdridptr[MAXID];
// static char xdrmodes[MAXID];
thread_local static int *ip = nullptr;
thread_local static int *buf = nullptr;

/*___________________________________________________________________________
 |
 | what follows are the C routines for opening, closing xdr streams
 | and the routine to read/write compressed coordinates together
 | with some routines to assist in this task (those are marked
 | static and cannot be called from user programs)
*/
#define MAXABS (float)(INT_MAX-2)

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif
static int magicints[] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0,
  8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
  80, 101, 128, 161, 203, 256, 322, 406, 512, 645,
  812, 1024, 1290, 1625, 2048, 2580, 3250, 4096, 5060, 6501,
  8192, 10321, 13003, 16384, 20642, 26007, 32768, 41285, 52015, 65536,
  82570, 104031, 131072, 165140, 208063, 262144, 330280, 416127,
  524287, 660561,
  832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
  4194304, 5284491, 6658042,
  8388607, 10568983, 13316085, 16777216 };

#define FIRSTIDX 9
/* note that magicints[FIRSTIDX-1] == 0 */
#define LASTIDX (sizeof(magicints) / sizeof(int) - 1 )

/*__________________________________________________________________________
 |
 | xdropen - open xdr file
 |
 | This versions differs from xdrstdio_create, because I need to know
 | the state of the file (read or write) so I can use xdr3dfcoord
 | in either read or write mode, and the file descriptor
 | so I can close the file (something xdr_destroy doesn't do).
 |
*/

XDR* xdropen( char * filename, const char * type)
{
  enum xdr_op lmode;
  
  if (*type == 'w' || *type == 'a') {
    lmode = XDR_ENCODE;
  } else {
    lmode = XDR_DECODE;
  }
  FILE* file = fopen(filename,type);
  if (file == nullptr) {
    throw std::runtime_error("could not open file");
  }
  XDR* xdr = new XDR;
  xdrstdio_create(xdr, file, lmode);
  return xdr;
}

/*_________________________________________________________________________
 |
 | xdrclose - close a xdr file
 |
 | This will flush the xdr buffers, and destroy the xdr stream.
 | It also closes the associated file descriptor (this is *not*
 | done by xdr_destroy).
 |
*/

int xdrclose(XDR *xdr)
{
  if (xdr == nullptr) {
    fprintf(stderr, "xdrclose: passed a NULL pointer\n");
    exit(1);
  }
  xdr_destroy(xdr);
  return fclose((FILE*)xdr->x_private);
}

/*_________________________________________________________________________
 |
 | xdrfreebuf - free the buffers used by xdr3dfcoord
 |
*/
void xdrfreebuf()
{
  if (ip) free(ip);
  if (buf) free(buf);
  ip = nullptr;
  buf = nullptr;
}


/*____________________________________________________________________________
 |
 | sendbits - encode num into buf using the specified number of bits
 |
 | This routines appends the value of num to the bits already present in
 | the array buf. You need to give it the number of bits to use and you
 | better make sure that this number of bits is enough to hold the value
 | Also num must be positive.
 |
*/

static void sendbits(int buf[], int num_of_bits, int num)
{
  unsigned int cnt, lastbyte;
  int lastbits;
  unsigned char * cbuf;

  cbuf = ((unsigned char *)buf) + 3 * sizeof(*buf);
  cnt = (unsigned int) buf[0];
  lastbits = buf[1];
  lastbyte =(unsigned int) buf[2];
  while (num_of_bits >= 8) {
    lastbyte = (lastbyte << 8) | ((num >> (num_of_bits -8)) /* & 0xff*/);
    cbuf[cnt++] = lastbyte >> lastbits;
    num_of_bits -= 8;
  }
  if (num_of_bits > 0) {
    lastbyte = (lastbyte << num_of_bits) | num;
    lastbits += num_of_bits;
    if (lastbits >= 8) {
      lastbits -= 8;
      cbuf[cnt++] = lastbyte >> lastbits;
    }
  }
  buf[0] = cnt;
  buf[1] = lastbits;
  buf[2] = lastbyte;
  if (lastbits>0) {
    cbuf[cnt] = lastbyte << (8 - lastbits);
  }
}

/*_________________________________________________________________________
 |
 | sizeofint - calculate bitsize of an integer
 |
 | return the number of bits needed to store an integer with given max size
 |
*/

static int sizeofint(const int size)
{
  unsigned int num = 1;
  int num_of_bits = 0;

  while (size >= (int) num && num_of_bits < 32) {
    num_of_bits++;
    num <<= 1;
  }
  return num_of_bits;
}

/*___________________________________________________________________________
 |
 | sizeofints - calculate 'bitsize' of compressed ints
 |
 | given the number of small unsigned integers and the maximum value
 | return the number of bits needed to read or write them with the
 | routines receiveints and sendints. You need this parameter when
 | calling these routines. Note that for many calls I can use
 | the variable 'smallidx' which is exactly the number of bits, and
 | So I don't need to call 'sizeofints for those calls.
*/

static int sizeofints( const int num_of_ints, unsigned int sizes[])
{
  int i, num;
  unsigned int num_of_bytes, num_of_bits, bytes[32], bytecnt, tmp;
  num_of_bytes = 1;
  bytes[0] = 1;
  num_of_bits = 0;
  for (i=0; i < num_of_ints; i++) {
    tmp = 0;
    for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
      tmp = bytes[bytecnt] * sizes[i] + tmp;
      bytes[bytecnt] = tmp & 0xff;
      tmp >>= 8;
    }
    while (tmp != 0) {
      bytes[bytecnt++] = tmp & 0xff;
      tmp >>= 8;
    }
    num_of_bytes = bytecnt;
  }
  num = 1;
  num_of_bytes--;
  while ((int)bytes[num_of_bytes] >= num) {
    num_of_bits++;
    num *= 2;
  }
  return num_of_bits + num_of_bytes * 8;
}

/*____________________________________________________________________________
 |
 | sendints - send a small set of small integers in compressed
 |
 | this routine is used internally by xdr3dfcoord, to send a set of
 | small integers to the buffer.
 | Multiplication with fixed (specified maximum ) sizes is used to get
 | to one big, multibyte integer. Although the routine could be
 | modified to handle sizes bigger than 16777216, or more than just
 | a few integers, this is not done, because the gain in compression
 | isn't worth the effort. Note that overflowing the multiplication
 | or the byte buffer (32 bytes) is unchecked and causes bad results.
 |
 */

static void sendints(int buf[], const int num_of_ints, const int num_of_bits,
                     unsigned int sizes[], unsigned int nums[])
{
  int i;
  unsigned int bytes[32], num_of_bytes, bytecnt, tmp;

  tmp = nums[0];
  num_of_bytes = 0;
  do {
    bytes[num_of_bytes++] = tmp & 0xff;
    tmp >>= 8;
  } while (tmp != 0);

  for (i = 1; i < num_of_ints; i++) {
    if (nums[i] >= sizes[i]) {
      fprintf(stderr,"major breakdown in sendints num %d doesn't "
              "match size %d\n", nums[i], sizes[i]);
      exit(1);
    }
    /* use one step multiply */
    tmp = nums[i];
    for (bytecnt = 0; bytecnt < num_of_bytes; bytecnt++) {
      tmp = bytes[bytecnt] * sizes[i] + tmp;
      bytes[bytecnt] = tmp & 0xff;
      tmp >>= 8;
    }
    while (tmp != 0) {
      bytes[bytecnt++] = tmp & 0xff;
      tmp >>= 8;
    }
    num_of_bytes = bytecnt;
  }
  if (num_of_bits >= (int)num_of_bytes * 8) {
    for (i = 0; i < (int)num_of_bytes; i++) {
      sendbits(buf, 8, bytes[i]);
    }
    sendbits(buf, num_of_bits - num_of_bytes * 8, 0);
  } else {
    for (i = 0; i < (int)num_of_bytes-1; i++) {
      sendbits(buf, 8, bytes[i]);
    }
    sendbits(buf, num_of_bits- (num_of_bytes -1) * 8, bytes[i]);
  }
}

/*___________________________________________________________________________
 |
 | receivebits - decode number from buf using specified number of bits
 |
 | extract the number of bits from the array buf and construct an integer
 | from it. Return that value.
 |
*/

static int receivebits(int buf[], int num_of_bits)
{
  int cnt, num;
  unsigned int lastbits, lastbyte;
  unsigned char * cbuf;
  int mask = (1 << num_of_bits) -1;

  cbuf = ((unsigned char *)buf) + 3 * sizeof(*buf);
  cnt = buf[0];
  lastbits = (unsigned int) buf[1];
  lastbyte = (unsigned int) buf[2];

  num = 0;
  while (num_of_bits >= 8) {
    lastbyte = ( lastbyte << 8 ) | cbuf[cnt++];
    num |=  (lastbyte >> lastbits) << (num_of_bits - 8);
    num_of_bits -=8;
  }
  if (num_of_bits > 0) {
    if ((int)lastbits < num_of_bits) {
      lastbits += 8;
      lastbyte = (lastbyte << 8) | cbuf[cnt++];
    }
    lastbits -= num_of_bits;
    num |= (lastbyte >> lastbits) & ((1 << num_of_bits) -1);
  }
  num &= mask;
  buf[0] = cnt;
  buf[1] = lastbits;
  buf[2] = lastbyte;
  return num;
}

/*____________________________________________________________________________
 |
 | receiveints - decode 'small' integers from the buf array
 |
 | this routine is the inverse from sendints() and decodes the small integers
 | written to buf by calculating the remainder and doing divisions with
 | the given sizes[]. You need to specify the total number of bits to be
 | used from buf in num_of_bits.
 |
*/

static void receiveints(int buf[], const int num_of_ints, int num_of_bits,
                        unsigned int sizes[], int nums[])
{
  int bytes[32];
  int i, j, num_of_bytes, p, num;

  bytes[1] = bytes[2] = bytes[3] = 0;
  num_of_bytes = 0;
  while (num_of_bits > 8) {
    bytes[num_of_bytes++] = receivebits(buf, 8);
    num_of_bits -= 8;
  }
  if (num_of_bits > 0) {
    bytes[num_of_bytes++] = receivebits(buf, num_of_bits);
  }
  for (i = num_of_ints-1; i > 0; i--) {
    num = 0;
    for (j = num_of_bytes-1; j >= 0; j--) {
      num = (num << 8) | bytes[j];
      p = num / sizes[i];
      bytes[j] = p;
      num = num - p * sizes[i];
    }
    nums[i] = num;
  }
  nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
}


/*____________________________________________________________________________
 |
 | xdr_header - read or write header
 |
*/
bool_t xdr_header(XDR *xd,int *n,int *ntimestep,float *time_value,float *box)
{ 
  int tmp = XTC_MAGIC;
  bool_t ok = true;
  ok&=xdr_int_check(xd,&tmp);
  ok&=xdr_int(xd,n);
  ok&=xdr_int(xd,ntimestep);
  ok&=xdr_float(xd,time_value);

  for (size_t i = 0; i < 9; i++)
  {
    /* code */
    ok&=xdr_float(xd,&(box[i]));
  }
  return ok;
}

/*____________________________________________________________________________
 |
 | xdr3dfcoord - read or write compressed 3d coordinates to xdr file.
 |
 | this routine reads or writes (depending on how you opened the file with
 | xdropen() ) a large number of 3d coordinates (stored in *fp).
 | The number of coordinates triplets to write is given by *size. On
 | read this number may be zero, in which case it reads as many as were written
 | or it may specify the number if triplets to read (which should match the
 | number written).
 | Compression is achieved by first converting all floating numbers to integer
 | using multiplication by *precision and rounding to the nearest integer.
 | Then the minimum and maximum value are calculated to determine the range.
 | The limited range of integers so found, is used to compress the coordinates.
 | In addition the differences between successive coordinates is calculated.
 | If the difference happens to be 'small' then only the difference is saved,
 | compressing the data even more. The notion of 'small' is changed dynamically
 | and is enlarged or reduced whenever needed or possible.
 | Extra compression is achieved in the case of GROMOS and coordinates of
 | water molecules. GROMOS first writes out the Oxygen position, followed by
 | the two hydrogens. In order to make the differences smaller (and thereby
 | compression the data better) the order is changed into first one hydrogen
 | then the oxygen, followed by the other hydrogen. This is rather special, but
 | it shouldn't harm in the general case.
 |
 */

int xdr3dfcoord(XDR *xdrs, float *fp, int *size, float *precision)
{
  static int oldsize;

  int minint[3], maxint[3], mindiff, *lip, diff;
  int lint1, lint2, lint3, oldlint1, oldlint2, oldlint3, smallidx;
  int minidx, maxidx;
  unsigned sizeint[3], sizesmall[3], bitsizeint[3], size3, *luip;
  int flag, k;
  int small, smaller, larger, i, is_small, is_smaller, run, prevrun;
  float *lfp, lf;
  int tmp, *thiscoord,  prevcoord[3];
  unsigned int tmpcoord[30];
  float tmpfloatcoord[9];

  int bufsize, xdrid, lsize;
  unsigned int bitsize;
  float inv_precision;
  int errval = 1;

  
  if (xdrs->x_op == XDR_ENCODE) {

    /* xdrs is open for writing */

    if (xdr_int(xdrs, size) == 0)
      return 0;
    size3 = *size * 3;
    /* when the number of coordinates is small, don't try to compress; just
     * write them as floats using xdr_vector
     */
    if (*size <= 9) {
      return (xdr_vector(xdrs, (char *) fp, size3, sizeof(*fp),
                         (xdrproc_t)xdr_float));
    }

    xdr_float(xdrs, precision);
    if (ip == nullptr) {
      ip = (int *) malloc(size3 * sizeof(*ip));
      if (ip == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      bufsize = (int) (size3 * 1.2);
      buf = (int *) malloc(bufsize * sizeof(*buf));
      if (buf == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      oldsize = *size;
    } else if (*size > oldsize) {
      ip = (int *) realloc(ip, size3 * sizeof(*ip));
      if (ip == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      bufsize = (int) (size3 * 1.2);
      buf = (int *) realloc(buf, bufsize * sizeof(*buf));
      if (buf == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      oldsize = *size;
    }
    /* buf[0-2] are special and do not contain actual data */
    buf[0] = buf[1] = buf[2] = 0;
    minint[0] = minint[1] = minint[2] = INT_MAX;
    maxint[0] = maxint[1] = maxint[2] = INT_MIN;
    prevrun = -1;
    lfp = fp;
    lip = ip;
    mindiff = INT_MAX;
    oldlint1 = oldlint2 = oldlint3 = 0;
    while (lfp < fp + size3) {
      /* find nearest integer */
      if (*lfp >= 0.0)
        lf = *lfp * *precision + 0.5;
      else
        lf = *lfp * *precision - 0.5;
      if (fabs(lf) > MAXABS) {
        /* scaling would cause overflow */
        errval = 0;
      }
      lint1 = (int) lf;
      if (lint1 < minint[0]) minint[0] = lint1;
      if (lint1 > maxint[0]) maxint[0] = lint1;
      *lip++ = lint1;
      lfp++;
      if (*lfp >= 0.0)
        lf = *lfp * *precision + 0.5;
      else
        lf = *lfp * *precision - 0.5;
      if (fabs(lf) > MAXABS) {
        /* scaling would cause overflow */
        errval = 0;
      }
      lint2 = (int) lf;
      if (lint2 < minint[1]) minint[1] = lint2;
      if (lint2 > maxint[1]) maxint[1] = lint2;
      *lip++ = lint2;
      lfp++;
      if (*lfp >= 0.0)
        lf = *lfp * *precision + 0.5;
      else
        lf = *lfp * *precision - 0.5;
      if (fabs(lf) > MAXABS) {
        /* scaling would cause overflow */
        errval = 0;
      }
      lint3 = (int) lf;
      if (lint3 < minint[2]) minint[2] = lint3;
      if (lint3 > maxint[2]) maxint[2] = lint3;
      *lip++ = lint3;
      lfp++;
      diff = abs(oldlint1-lint1)+abs(oldlint2-lint2)+abs(oldlint3-lint3);
      if (diff < mindiff && lfp > fp + 3)
        mindiff = diff;
      oldlint1 = lint1;
      oldlint2 = lint2;
      oldlint3 = lint3;
    }
    xdr_int(xdrs, &(minint[0]));
    xdr_int(xdrs, &(minint[1]));
    xdr_int(xdrs, &(minint[2]));

    xdr_int(xdrs, &(maxint[0]));
    xdr_int(xdrs, &(maxint[1]));
    xdr_int(xdrs, &(maxint[2]));

    if ((float)maxint[0] - (float)minint[0] >= MAXABS ||
        (float)maxint[1] - (float)minint[1] >= MAXABS ||
        (float)maxint[2] - (float)minint[2] >= MAXABS) {
      /* turning value in unsigned by subtracting minint
       * would cause overflow
       */
      errval = 0;
    }
    sizeint[0] = maxint[0] - minint[0]+1;
    sizeint[1] = maxint[1] - minint[1]+1;
    sizeint[2] = maxint[2] - minint[2]+1;

    /* check if one of the sizes is to big to be multiplied */
    if ((sizeint[0] | sizeint[1] | sizeint[2] ) > 0xffffff) {
      bitsizeint[0] = sizeofint(sizeint[0]);
      bitsizeint[1] = sizeofint(sizeint[1]);
      bitsizeint[2] = sizeofint(sizeint[2]);
      bitsize = 0; /* flag the use of large sizes */
    } else {
      bitsize = sizeofints(3, sizeint);
    }
    lip = ip;
    luip = (unsigned int *) ip;
    smallidx = FIRSTIDX;
    while (smallidx < (int)LASTIDX && magicints[smallidx] < mindiff) {
      smallidx++;
    }
    xdr_int(xdrs, &smallidx);
    maxidx = MYMIN((int)LASTIDX, smallidx + 8) ;
    minidx = maxidx - 8; /* often this equal smallidx */
    smaller = magicints[MYMAX(FIRSTIDX, smallidx-1)] / 2;
    small = magicints[smallidx] / 2;
    sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx];
    larger = magicints[maxidx] / 2;
    i = 0;
    while (i < *size) {
      is_small = 0;
      thiscoord = (int *)(luip) + i * 3;
      if (smallidx < maxidx && i >= 1 &&
          abs(thiscoord[0] - prevcoord[0]) < larger &&
          abs(thiscoord[1] - prevcoord[1]) < larger &&
          abs(thiscoord[2] - prevcoord[2]) < larger) {
        is_smaller = 1;
      } else if (smallidx > minidx) {
        is_smaller = -1;
      } else {
        is_smaller = 0;
      }
      if (i + 1 < *size) {
        if (abs(thiscoord[0] - thiscoord[3]) < small &&
            abs(thiscoord[1] - thiscoord[4]) < small &&
            abs(thiscoord[2] - thiscoord[5]) < small) {
          /* interchange first with second atom for better
           * compression of water molecules
           */
          tmp = thiscoord[0]; thiscoord[0] = thiscoord[3];
          thiscoord[3] = tmp;
          tmp = thiscoord[1]; thiscoord[1] = thiscoord[4];
          thiscoord[4] = tmp;
          tmp = thiscoord[2]; thiscoord[2] = thiscoord[5];
          thiscoord[5] = tmp;
          is_small = 1;
        }

      }
      tmpcoord[0] = thiscoord[0] - minint[0];
      tmpcoord[1] = thiscoord[1] - minint[1];
      tmpcoord[2] = thiscoord[2] - minint[2];
      if (bitsize == 0) {
        sendbits(buf, bitsizeint[0], tmpcoord[0]);
        sendbits(buf, bitsizeint[1], tmpcoord[1]);
        sendbits(buf, bitsizeint[2], tmpcoord[2]);
      } else {
        sendints(buf, 3, bitsize, sizeint, tmpcoord);
      }
      prevcoord[0] = thiscoord[0];
      prevcoord[1] = thiscoord[1];
      prevcoord[2] = thiscoord[2];
      thiscoord = thiscoord + 3;
      i++;

      run = 0;
      if (is_small == 0 && is_smaller == -1)
        is_smaller = 0;
      while (is_small && run < 8*3) {
        if (is_smaller == -1 && (SQR(thiscoord[0] - prevcoord[0]) +
                                 SQR(thiscoord[1] - prevcoord[1]) +
                                 SQR(thiscoord[2] - prevcoord[2]) >=
                                 smaller * smaller)) {
          is_smaller = 0;
        }

        tmpcoord[run++] = thiscoord[0] - prevcoord[0] + small;
        tmpcoord[run++] = thiscoord[1] - prevcoord[1] + small;
        tmpcoord[run++] = thiscoord[2] - prevcoord[2] + small;

        prevcoord[0] = thiscoord[0];
        prevcoord[1] = thiscoord[1];
        prevcoord[2] = thiscoord[2];

        i++;
        thiscoord = thiscoord + 3;
        is_small = 0;
        if (i < *size &&
            abs(thiscoord[0] - prevcoord[0]) < small &&
            abs(thiscoord[1] - prevcoord[1]) < small &&
            abs(thiscoord[2] - prevcoord[2]) < small) {
          is_small = 1;
        }
      }
      if (run != prevrun || is_smaller != 0) {
        prevrun = run;
        sendbits(buf, 1, 1); /* flag the change in run-length */
        sendbits(buf, 5, run+is_smaller+1);
      } else {
        sendbits(buf, 1, 0); /* flag the fact that runlength did not change */
      }
      for (k=0; k < run; k+=3) {
        sendints(buf, 3, smallidx, sizesmall, &tmpcoord[k]);
      }
      if (is_smaller != 0) {
        smallidx += is_smaller;
        if (is_smaller < 0) {
          small = smaller;
          smaller = magicints[smallidx-1] / 2;
        } else {
          smaller = small;
          small = magicints[smallidx] / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx];
      }
    }
    if (buf[1] != 0) buf[0]++;
    xdr_int(xdrs, &(buf[0])); /* buf[0] holds the length in bytes */
    return errval * (xdr_opaque(xdrs, (caddr_t)&(buf[3]), (u_int)buf[0]));
  } else {

    /* xdrs is open for reading */

    if (xdr_int(xdrs, &lsize) == 0)
      return 0;
    if (*size != 0 && lsize != *size) {
      return 0;
      // fprintf(stderr, "wrong number of coordinates in xdr3dfcoor; "
              // "%d arg vs %d in file", *size, lsize);
    }
    *size = lsize;
    size3 = *size * 3;
    if (*size <= 9) {
      return (xdr_vector(xdrs, (char *) (fp==nullptr?&tmpfloatcoord[0]:fp), size3, sizeof(*fp),
                         (xdrproc_t)xdr_float));
    }
    xdr_float(xdrs, precision);
    if (ip == nullptr) {
      ip = (int *) malloc(size3 * sizeof(*ip));
      if (ip == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      bufsize = (int) (size3 * 1.2);
      buf = (int *) malloc(bufsize * sizeof(*buf));
      if (buf == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      oldsize = *size;
    } else if (*size > oldsize) {
      ip = (int *)realloc(ip, size3 * sizeof(*ip));
      if (ip == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      bufsize = (int) (size3 * 1.2);
      buf = (int *)realloc(buf, bufsize * sizeof(*buf));
      if (buf == nullptr) {
        fprintf(stderr,"malloc failed\n");
        exit(1);
      }
      oldsize = *size;
    }
    buf[0] = buf[1] = buf[2] = 0;

    xdr_int(xdrs, &(minint[0]));
    xdr_int(xdrs, &(minint[1]));
    xdr_int(xdrs, &(minint[2]));

    xdr_int(xdrs, &(maxint[0]));
    xdr_int(xdrs, &(maxint[1]));
    xdr_int(xdrs, &(maxint[2]));

    sizeint[0] = maxint[0] - minint[0]+1;
    sizeint[1] = maxint[1] - minint[1]+1;
    sizeint[2] = maxint[2] - minint[2]+1;

    /* check if one of the sizes is to big to be multiplied */
    if ((sizeint[0] | sizeint[1] | sizeint[2] ) > 0xffffff) {
      bitsizeint[0] = sizeofint(sizeint[0]);
      bitsizeint[1] = sizeofint(sizeint[1]);
      bitsizeint[2] = sizeofint(sizeint[2]);
      bitsize = 0; /* flag the use of large sizes */
    } else {
      bitsize = sizeofints(3, sizeint);
    }

    xdr_int(xdrs, &smallidx);
    maxidx = MYMIN((int)LASTIDX, smallidx + 8) ;
    minidx = maxidx - 8; /* often this equal smallidx */
    smaller = magicints[MYMAX(FIRSTIDX, smallidx-1)] / 2;
    small = magicints[smallidx] / 2;
    sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx] ;
    larger = magicints[maxidx];

    /* buf[0] holds the length in bytes */

    if (xdr_int(xdrs, &(buf[0])) == 0)
      return 0;
    if (fp==nullptr){
      auto cnt=(u_int)buf[0];
      auto rndup = cnt % BYTES_PER_XDR_UNIT;
      if (rndup > 0)
        rndup = BYTES_PER_XDR_UNIT - rndup;
      return xdr_setpos(xdrs,xdr_getpos(xdrs)+cnt+rndup);
    }
    if (xdr_opaque(xdrs, (caddr_t)&(buf[3]), (u_int)buf[0]) == 0)
      return 0;
    buf[0] = buf[1] = buf[2] = 0;

    lfp = fp;
    inv_precision = 1.0 / * precision;
    run = 0;
    i = 0;
    lip = ip;
    while (i < lsize) {
      thiscoord = (int *)(lip) + i * 3;

      if (bitsize == 0) {
        thiscoord[0] = receivebits(buf, bitsizeint[0]);
        thiscoord[1] = receivebits(buf, bitsizeint[1]);
        thiscoord[2] = receivebits(buf, bitsizeint[2]);
      } else {
        receiveints(buf, 3, bitsize, sizeint, thiscoord);
      }

      i++;
      thiscoord[0] += minint[0];
      thiscoord[1] += minint[1];
      thiscoord[2] += minint[2];

      prevcoord[0] = thiscoord[0];
      prevcoord[1] = thiscoord[1];
      prevcoord[2] = thiscoord[2];


      flag = receivebits(buf, 1);
      is_smaller = 0;
      if (flag == 1) {
        run = receivebits(buf, 5);
        is_smaller = run % 3;
        run -= is_smaller;
        is_smaller--;
      }
      if (run > 0) {
        thiscoord += 3;
        for (k = 0; k < run; k+=3) {
          receiveints(buf, 3, smallidx, sizesmall, thiscoord);
          i++;
          thiscoord[0] += prevcoord[0] - small;
          thiscoord[1] += prevcoord[1] - small;
          thiscoord[2] += prevcoord[2] - small;
          if (k == 0) {
            /* interchange first with second atom for better
             * compression of water molecules
             */
            tmp = thiscoord[0]; thiscoord[0] = prevcoord[0];
            prevcoord[0] = tmp;
            tmp = thiscoord[1]; thiscoord[1] = prevcoord[1];
            prevcoord[1] = tmp;
            tmp = thiscoord[2]; thiscoord[2] = prevcoord[2];
            prevcoord[2] = tmp;
            *lfp++ = prevcoord[0] * inv_precision;
            *lfp++ = prevcoord[1] * inv_precision;
            *lfp++ = prevcoord[2] * inv_precision;
          } else {
            prevcoord[0] = thiscoord[0];
            prevcoord[1] = thiscoord[1];
            prevcoord[2] = thiscoord[2];
          }
          *lfp++ = thiscoord[0] * inv_precision;
          *lfp++ = thiscoord[1] * inv_precision;
          *lfp++ = thiscoord[2] * inv_precision;
        }
      } else {
        *lfp++ = thiscoord[0] * inv_precision;
        *lfp++ = thiscoord[1] * inv_precision;
        *lfp++ = thiscoord[2] * inv_precision;
      }
      smallidx += is_smaller;
      if (is_smaller < 0) {
        small = smaller;
        if (smallidx > FIRSTIDX) {
          smaller = magicints[smallidx - 1] /2;
        } else {
          smaller = 0;
        }
      } else if (is_smaller > 0) {
        smaller = small;
        small = magicints[smallidx] / 2;
      }
      sizesmall[0] = sizesmall[1] = sizesmall[2] = magicints[smallidx] ;
    }
  }
  return 1;
}

unsigned int xdrgetpos(XDR *xdrs)
{
  return xdr_getpos(xdrs);
}
bool_t xdrsetpos(XDR *xdrs,unsigned int* pos)
{
  return xdr_setpos(xdrs,*pos);
}