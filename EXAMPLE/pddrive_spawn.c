/*! \file
Copyright (c) 2003, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required 
approvals from U.S. Dept. of Energy) 

All rights reserved. 

The source code is distributed under BSD license, see the file License.txt
at the top-level directory.
*/


/*! @file 
 * \brief Driver program for PDGSSVX example
 *
 * <pre>
 * -- Distributed SuperLU routine (version 9.0) --
 * Lawrence Berkeley National Lab, Univ. of California Berkeley.
 * November 1, 2007
 * December 6, 2018
 * </pre>
 */

#include <math.h>
#include "superlu_ddefs.h"
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>


/*! \brief
 *
 * <pre>
 * Purpose
 * =======
 *
 * The driver program PDDRIVE.
 *
 * This example illustrates how to use PDGSSVX with the full
 * (default) options to solve a linear system.
 * 
 * Five basic steps are required:
 *   1. Initialize the MPI environment and the SuperLU process grid
 *   2. Set up the input matrix and the right-hand side
 *   3. Set the options argument
 *   4. Call pdgssvx
 *   5. Release the process grid and terminate the MPI environment
 *
 * With MPICH,  program may be run by typing:
 *    mpiexec -n <np> pddrive -r <proc rows> -c <proc columns> big.rua
 * </pre>
 */

void set_env_from_string(char *env_str) {
    char *line;
    char *saveptr1, *saveptr2;

    line = strtok_r(env_str, "\n", &saveptr1);
    while (line != NULL) {
        char *key = strtok_r(line, "=", &saveptr2);
        char *value = strtok_r(NULL, "=", &saveptr2);

        if (key && value) {
            setenv(key, value, 1);
        }
        
        line = strtok_r(NULL, "\n", &saveptr1);
    }
}

int main(int argc, char *argv[])
{
    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A;
    dScalePermstruct_t ScalePermstruct;
    dLUstruct_t LUstruct;
    dSOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
    double   *b, *xtrue;
    int    m, n;
    int      nprow, npcol,lookahead,colperm;
    int      iam, info, ldb, ldx, nrhs;
    char     **cpp, c, *postfix;;
    FILE *fp;
    int cpp_defs();
    int ii, omp_mpi_level;
	MPI_Comm parent;
	float result[2];
	result[0]=0.0;
	result[1]=0.0;
	
    nprow = 1;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs = 1;   /* Number of right-hand side. */
	lookahead = 10; 
	colperm = 4;

    MPI_Session session = MPI_SESSION_NULL; 
    MPI_Group group = MPI_GROUP_NULL;
    MPI_Comm union_comm = MPI_COMM_NULL;
    MPI_Info inf = MPI_INFO_NULL;
    char main_pset[MPI_MAX_PSET_NAME_LEN];
    int flag = 0;

    printf("\n\n\nwe are inside pddrive_spawn (Session Mode)\n\n\n\n");

    setenv("OMPI_MPI_THREAD_LEVEL", "MPI_THREAD_MULTIPLE", 1);

    int err;
    
    MPI_Info session_info = MPI_INFO_NULL;
    MPI_Info_create(&session_info);
    MPI_Info_set(session_info, "thread_level", "MPI_THREAD_MULTIPLE");
    
    err = MPI_Session_init(session_info, MPI_ERRORS_ARE_FATAL, &session);
    MPI_Info_free(&session_info);

    MPI_Info info_inside = MPI_INFO_NULL;
    MPI_Info_create(&info_inside);
    MPI_Session_get_info(session, &info_inside);

    char thread_str[MPI_MAX_PSET_NAME_LEN] = "";

    MPI_Info_get(info_inside, "thread_level", MPI_MAX_PSET_NAME_LEN, thread_str, &flag);
    printf("info inside says %s\n", thread_str);

    if (err != MPI_SUCCESS) {
        printf("Error initializing MPI Session\n");
        MPI_Abort(MPI_COMM_WORLD, err);
    }
    
    /* Query what threading level we got */
    int temp_thread_level;
    MPI_Query_thread(&temp_thread_level);
    switch (temp_thread_level) {
      case MPI_THREAD_SINGLE:
		printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
		fflush(stdout);
	break;
      case MPI_THREAD_FUNNELED:
		printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
		fflush(stdout);
	break;
      case MPI_THREAD_SERIALIZED:
		printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
		fflush(stdout);
	break;
      case MPI_THREAD_MULTIPLE:
		printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
		fflush(stdout);
	break;
    }



    strcpy(main_pset, "mpi://WORLD");
    printf("test1\n");

    const char *keys[2] = {"inter_pset", "env_str"};

    // Pass 'keys' directly. In C, the array name 'keys' evaluates to a char**
    MPI_Session_get_pset_data(session, main_pset, main_pset, keys, 2, 1, &inf);

    char env_str[MPI_MAX_PSET_NAME_LEN] = "";

    /* CLEANUP AFTER USE */
    if (inf != MPI_INFO_NULL) {
        MPI_Info_get(inf, "inter_pset", MPI_MAX_PSET_NAME_LEN, main_pset, &flag);
        MPI_Info_get(inf, "env_str", MPI_MAX_PSET_NAME_LEN, env_str, &flag);
        MPI_Info_free(&inf);
    }
    //free(keys[0]);

    // Set environment variables from the retrieved env_str
    set_env_from_string(env_str);
    printf("nrel is now: %s\n", getenv("NREL"));
    

    //MPI_Info_get(inf, "inter_pset", MPI_MAX_PSET_NAME_LEN, main_pset, &flag);

    printf("test2\n");

    printf("Using pset: %s\n", main_pset);
   
    /* Create Group and Communicator from PSet */
    MPI_Group_from_session_pset (session, main_pset, &group);
    MPI_Comm_create_from_group(group, "mpi.forum.example", MPI_INFO_NULL, MPI_ERRORS_ARE_FATAL, &union_comm);



    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Group group2 = MPI_GROUP_NULL;

    MPI_Group_from_session_pset(session, "mpi://WORLD", &group2);
    MPI_Comm_create_from_group(group2, "lcm.example", MPI_INFO_NULL, MPI_ERRORS_RETURN, &comm);

    printf("after comm creation\n");

    /*
    group2 = session.Create_group("mpi://WORLD")
    local_comm = MPI.Intracomm.Create_from_group(
        group2, "lcm.example", MPI.INFO_NULL, MPI.ERRORS_RETURN
    )
    */

    printf("test3\n");
    
    //parent = MPI_COMM_NULL;
	
	

#if ( VAMPIR>=1 )
    VT_traceoff(); 
#endif

#if ( VTUNE>=1 )
	__itt_pause();
#endif
	
    /* Parse command line argv[]. */
    for (cpp = argv+1; *cpp; ++cpp) {
	if ( **cpp == '-' ) {
	    c = *(*cpp+1);
	    ++cpp;
	    switch (c) {
	      case 'h':
		  printf("Options:\n");
		  printf("\t-r <int>: process rows    (default %4d)\n", nprow);
		  printf("\t-c <int>: process columns (default %4d)\n", npcol);
		  exit(0);
		  break;
	      case 'r': nprow = atoi(*cpp);
		        break;
	      case 'c': npcol = atoi(*cpp);
		        break;
	      case 'l': lookahead = atoi(*cpp);
		        break;
	      case 'p': colperm = atoi(*cpp);
		        break;					
	    }
	} else { /* Last arg is considered a filename */
	    if ( !(fp = fopen(*cpp, "r")) ) {
                ABORT("File does not exist");
            }
	    break;
	}
    }

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID. 
       ------------------------------------------------------------*/
    superlu_gridinit(comm, nprow, npcol, &grid);
	
    if(grid.iam==0){
	MPI_Query_thread(&omp_mpi_level);
    switch (omp_mpi_level) {
      case MPI_THREAD_SINGLE:
		printf("MPI_Query_thread with MPI_THREAD_SINGLE\n");
		fflush(stdout);
	break;
      case MPI_THREAD_FUNNELED:
		printf("MPI_Query_thread with MPI_THREAD_FUNNELED\n");
		fflush(stdout);
	break;
      case MPI_THREAD_SERIALIZED:
		printf("MPI_Query_thread with MPI_THREAD_SERIALIZED\n");
		fflush(stdout);
	break;
      case MPI_THREAD_MULTIPLE:
		printf("MPI_Query_thread with MPI_THREAD_MULTIPLE\n");
		fflush(stdout);
	break;
    }
	}
	
    /* Bail out if I do not belong in the grid. */
    iam = grid.iam;
    if ( iam == -1 )	goto out;
    if ( !iam ) {
	int v_major, v_minor, v_bugfix;
#ifdef __INTEL_COMPILER
	printf("__INTEL_COMPILER is defined\n");
#endif
	printf("__STDC_VERSION__ %ld\n", __STDC_VERSION__);

	superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
	printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

	printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid:\t\t%d X %d\n", (int)grid.nprow, (int)grid.npcol);
	fflush(stdout);
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    for(ii = 0;ii<strlen(*cpp);ii++){
	if((*cpp)[ii]=='.'){
		postfix = &((*cpp)[ii+1]);
	}
    }
    // printf("%s\n", postfix);
	
    /* ------------------------------------------------------------
       GET THE MATRIX FROM FILE AND SETUP THE RIGHT HAND SIDE. 
       ------------------------------------------------------------*/
    dcreate_matrix_postfix(&A, nrhs, &b, &ldb, &xtrue, &ldx, fp, postfix, &grid);
    if ( !(berr = doubleMalloc_dist(nrhs)) )
	ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       NOW WE SOLVE THE LINEAR SYSTEM.
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES;
		options.DiagInv       = NO;
     */
    set_default_options_dist(&options);
	options.IterRefine = NOREFINE;
	options.DiagInv = YES;							   
#if 0
    options.RowPerm = NOROWPERM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO; 
    options.ReplaceTinyPivot = YES;
#endif


	options.ColPerm           = colperm;
	options.num_lookaheads           = lookahead;


    if (!iam) {
	//print_sp_ienv_dist(&options);
	print_options_dist(&options);
	fflush(stdout);
    }

    m = A.nrow;
    n = A.ncol;

    dScalePermstructInit(m, n, &ScalePermstruct);
    dLUstructInit(n, &LUstruct);

    /* Initialize the statistics variables. */
    PStatInit(&stat);

    /* Call the linear equation solver. */
    pdgssvx(&options, &A, &ScalePermstruct, b, ldb, nrhs, &grid,
	    &LUstruct, &SOLVEstruct, berr, &stat, &info);
    

    /* Check the accuracy of the solution. */
    pdinf_norm_error(iam, ((NRformat_loc *)A.Store)->m_loc,
		     nrhs, b, ldb, xtrue, ldx, grid.comm);
    
    PStatPrint(&options, &stat, &grid);        /* Print the statistics. */
    
	
	
	
	/* sending the results (numerical factorization time and total memory) to the parent process */ 
	    float total;
		superlu_dist_mem_usage_t num_mem_usage;
	    dQuerySpace_dist(n, &LUstruct, &grid, &stat, &num_mem_usage);
	    MPI_Allreduce( &num_mem_usage.total, &total,
		       1, MPI_FLOAT, MPI_SUM, grid.comm );
			   
		result[0] = stat.utime[FACT];   
		result[1] = total * 1e-6;     
		if (!iam) {
			printf("returning data:\n"
		   "    Factor time :        %10.4f\n    Total MEM : %10.4f\n",
		   stat.utime[FACT], total * 1e-6);
			printf("**************************************************\n");
			fflush(stdout);
		}	
	
	
	
		//MPI_Bcast(result,2,MPI_FLOAT,0,parent);
		
    		//if (!iam) {
                 //   MPI_Send( result, 2, MPI_FLOAT, 0,
                  //           2, &parent);
    	//	}

	
	
    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    PStatFree(&stat);
    Destroy_CompRowLoc_Matrix_dist(&A);
    dScalePermstructFree(&ScalePermstruct);
    dDestroy_LU(n, &grid, &LUstruct);
    dLUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        dSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(b);
    SUPERLU_FREE(xtrue);
    SUPERLU_FREE(berr);

    printf("we reached here\n");
    fflush(stdout);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
out:
    /* Parent communication removed */
    /* if(parent!=MPI_COMM_NULL)
	MPI_Reduce(result, MPI_BOTTOM, 2, MPI_FLOAT,MPI_MAX, 0, parent); */
    MPI_Reduce(result, MPI_BOTTOM, 2, MPI_FLOAT,MPI_MAX, 0, union_comm);
    superlu_gridexit(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
	   
    /* if(parent!=MPI_COMM_NULL)
	MPI_Comm_disconnect(&parent); */

    printf("we reached here2\n");
    fflush(stdout);

    MPI_Comm_disconnect(&union_comm);

    printf("we reached here2.1\n");
    fflush(stdout);

    


    MPI_Session_finalize(&session);

    printf("we reached here3\n");
    fflush(stdout);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

}


int cpp_defs()
{
    printf(".. CPP definitions:\n");
#if ( PRNTlevel>=1 )
    printf("\tPRNTlevel = %d\n", PRNTlevel);
#endif
#if ( DEBUGlevel>=1 )
    printf("\tDEBUGlevel = %d\n", DEBUGlevel);
#endif
#if ( PROFlevel>=1 )
    printf("\tPROFlevel = %d\n", PROFlevel);
#endif
#if ( StaticPivot>=1 )
    printf("\tStaticPivot = %d\n", StaticPivot);
#endif
    printf("....\n");
    return 0;
}
