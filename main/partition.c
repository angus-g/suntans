/*
 * File: partition.c
 * Author: Oliver B. Fringer
 * Institution: Stanford University
 * --------------------------------
 * This file contains functions that use the ParMetis libraries
 *
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior 
 * University. All Rights Reserved.
 *
 */
#include "partition.h"
#include "memory.h"
#include "grid.h"

//Parmetis 2.0
//#include "parmetis.h"
//Parmetis 3.1
#include <parmetis.h>

typedef struct {
  int nvtx;

  int *vtxdist;
  int *xadj;
  int *vwgt;
  int *adjncy;
} graph_t;

// Private function
static void GetDistributedGraph(graph_t *graph, gridT *grid, MPI_Comm comm);

/*
 * Function: GetPartitioning
 * Usage: GetPartitioning(maingrid,localgrid,myproc,numprocs,comm);
 * ----------------------------------------------------------------
 * This function uses the ParMetis libraries to compute the grid partitioning and places
 * the partition number into the maingrid->part array.
 *
 */
void GetPartitioning(gridT *maingrid, gridT **localgrid, int myproc, int numprocs, MPI_Comm comm) {
  int numflag=0, wgtflag=0, options[5], edgecut, ncon, *nvtxs;
  float ub, *tpwgts;
  graph_t graph;
  MPI_Status status;

  if (numprocs > 1) {
    options[0] = 0; // no options
    wgtflag = 2; // weights on vertices only
    numflag = 0; // 0-indexing
    ncon = 1; // number of weights per vertex
    ub = 1.05f; // imbalance tolerance for weights

    tpwgts = malloc(numprocs * sizeof(float));
    for (int i = 0; i < numprocs; i++) {
      tpwgts[i] = 1.0f / (float)numprocs;
    }

    // distribute the full grid over all processes
    GetDistributedGraph(&graph, maingrid, comm);

    (*localgrid)->part = SunMalloc(graph.nvtx*sizeof(int),"GetPartitioning");

    /*
     * Partition the graph and create the part array.
     */
    if (myproc == 0 && VERBOSE > 2) printf("Partitioning with ParMETIS_PartKway...\n");

    ParMETIS_V3_PartKway(graph.vtxdist, graph.xadj, graph.adjncy, graph.vwgt, NULL,
			 &wgtflag, &numflag, &ncon, &numprocs, tpwgts, &ub,
			 options, &edgecut, (*localgrid)->part, &comm);

    if(myproc == 0 && VERBOSE > 2) printf("Redistributing the partition arrays...\n");
    nvtxs = malloc(numprocs * sizeof(int));
    for (int i = 0; i < numprocs; i++) {
      nvtxs[i] = graph.vtxdist[i + 1] - graph.vtxdist[i];
    }

    // let everybody know about the new distribution
    MPI_Allgatherv((void *)(*localgrid)->part, graph.nvtx, MPI_INT,
		maingrid->part, nvtxs, graph.vtxdist, MPI_INT, comm);

    SunFree((*localgrid)->part, graph.nvtx*sizeof(int),"GetPartitioning");

    free(tpwgts);
    free(nvtxs);

    free(graph.vtxdist);
    free(graph.xadj);
    free(graph.vwgt);
    free(graph.adjncy);
  } else {
    for(int j = 0; j < maingrid->Nc; j++)
      maingrid->part[j] = 0;
  }
}

static void GetDistributedGraph(graph_t *graph, gridT *grid, MPI_Comm comm)
{
  int numprocs, rank;

  // number of vertices distributed to each process (for scatter)
  int *pvtx = NULL, *pvtxp1 = NULL;
  // number of edges on each process, and their offset in the global adjncy
  int *pedg = NULL, *edge_offsets = NULL;

  MPI_Comm_size(comm, &numprocs);
  MPI_Comm_rank(comm, &rank);

  // allocate array to hold vertex distribution
  graph->vtxdist = malloc(sizeof(int) * (numprocs + 1));

  // construct vertex distribution
  if (rank == 0) {
    pvtx = malloc(sizeof(int) * numprocs);
    pvtxp1 = malloc(sizeof(int) * numprocs);
    pedg = malloc(sizeof(int) * numprocs);
    edge_offsets = malloc(sizeof(int) * numprocs);

    graph->vtxdist[0] = 0;
    edge_offsets[0] = 0;

    for (int i = 0, k = grid->Nc; i < numprocs; i++) {
      pvtx[i] = k / (numprocs - i); // number of vertices for this processor
      pvtxp1[i] = pvtx[i] + 1; // size of xadj for this processor

      graph->vtxdist[i+1] = graph->vtxdist[i] + pvtx[i];

      k -= pvtx[i];
    }

    for (int i = 0, k = 0; i < numprocs; i++) {
      pedg[i] = grid->xadj[k + pvtx[i]] - grid->xadj[k];
      k += pvtx[i];

      if (i < numprocs - 1) {
	edge_offsets[i+1] = edge_offsets[i] + pedg[i];
      }
    }
  }
  MPI_Bcast((void *)graph->vtxdist, numprocs + 1, MPI_INT, 0, comm);

  // number of local vertices
  int nvtx = graph->vtxdist[rank + 1] - graph->vtxdist[rank];
  graph->nvtx = nvtx;

  // allocate adjacency list counts and vertex weights
  graph->xadj = malloc(sizeof(int) * (nvtx + 1));
  graph->vwgt = malloc(sizeof(int) * nvtx);

  // send pvtx+1 elements of xadj to each processor
  MPI_Scatterv(grid->xadj, pvtxp1, graph->vtxdist, MPI_INT,
	       graph->xadj, nvtx + 1, MPI_INT, 0, comm);
  // convert to local vertex numbering
  for (int i = nvtx; i >= 0; i--) {
    graph->xadj[i] -= graph->xadj[0];
  }

  // send pvtx elements of vwgt to each processor
  MPI_Scatterv(grid->vwgt, pvtx, graph->vtxdist, MPI_INT,
	       graph->vwgt, nvtx, MPI_INT, 0, comm);

  // number of local edges
  int nedg = graph->xadj[nvtx];
  graph->adjncy = malloc(sizeof(int) * nedg);

  // scatter adjncy to each processor
  MPI_Scatterv(grid->adjncy, pedg, edge_offsets, MPI_INT,
	       graph->adjncy, nedg, MPI_INT, 0, comm);

  if (rank == 0) {
    free(pvtx);
    free(pvtxp1);
    free(pedg);
    free(edge_offsets);
  }
}
