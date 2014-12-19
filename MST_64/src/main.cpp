#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "RecMST.cuh"

////////////////////////////////////////////////
// Read the Graph in our format
////////////////////////////////////////////////
void ReadGraph(char *filename,
			   unsigned int **edge_list, 
			   unsigned int **vertex_list, 
			   unsigned int **weight_list,
			   unsigned int *no_of_vertices,
			   unsigned int *no_of_edges)
{
	printf("\nReading Graph...\n");
	FILE *fp;
	fp = fopen(filename,"r");
	fscanf(fp,"%d",no_of_vertices);
	*vertex_list = (unsigned int*)malloc(sizeof(unsigned int)*(*no_of_vertices));
	int start, index ;
	for (unsigned int i = 0 ; i < *no_of_vertices ; i++ )
	{
		fscanf(fp,"%d %d",&start, &index) ;
		(*vertex_list)[i] = start ;
	}
	unsigned int source = 0 ;
	fscanf(fp,"%d",&source);
	fscanf(fp,"%d",no_of_edges);
	*edge_list = (unsigned int*) malloc (sizeof(unsigned int)* (*no_of_edges));
	*weight_list = (unsigned int*) malloc (sizeof(unsigned int)*(*no_of_edges));
	unsigned int edgeindex, edgeweight ;
	for( unsigned int i = 0 ; i < *no_of_edges ; i++ )
	{
		fscanf(fp,"%d %d",&edgeindex, &edgeweight);
		if(edgeweight==0) 
			printf("Wieght is 0\n");
		(*edge_list)[i] = edgeindex ;
		(*weight_list)[i] =  edgeweight ;
	}
	fclose(fp);
	printf("File read successfully.\nNumber of Vertices: %d\nNumber of Edges: %d\n", *no_of_vertices, *no_of_edges);
}



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
// 	if(argc<2)
// 	{
// 		printf("Specify an Input Graph.\n");
// 		exit(1);
// 	}

	unsigned int *edge_list;
	unsigned int *vertex_list;
	unsigned int *weight_list;			
	bool *output_MST_list;				
	unsigned int no_of_vertices;	
	unsigned int no_of_edges;		

	ReadGraph(argv[1],&edge_list, &vertex_list, &weight_list, &no_of_vertices, &no_of_edges);
	//Allocate memory to test the output of the program
	output_MST_list = (bool*)malloc(sizeof(bool)*no_of_edges);

	MST mst;
	mst.Init(edge_list, vertex_list, weight_list, no_of_vertices, no_of_edges);
	mst.DoMST(output_MST_list,no_of_edges);
	mst.FreeMem();

	double totalmstedges=0;
	double totalmstweight=0;
	FILE *fp ;
	fp = fopen(argv[2],"w");
	if (fp == NULL) 
	{
		printf("can not write result file");
	}
	//printf("\n\nSelected Edges in MST...\n\n");
	for(unsigned int i=0;i<no_of_edges;i++)
		if(output_MST_list[i]==true)
		{
			int j = 0 ; 
			 for(j = 0 ; j <no_of_vertices ; j ++)
			 {
				 if(*(vertex_list+j) <=i && i< *(vertex_list+j+1))
					 break ;
			 }
			fprintf(fp,"%d %d %d\n",j+1, edge_list[i]+1,weight_list[i]);
			totalmstedges++;
			totalmstweight+=weight_list[i];
		}
		fprintf(fp,"%d\n",(int)totalmstedges);
		fprintf(fp,"%f",totalmstweight);
		fclose(fp);
		printf("write successfully...\n");
	    printf("\nNumber of edges in MST, must be equal to (%d-1): %d\nTotal MST weight: %f\n\n", no_of_vertices, (int )totalmstedges, totalmstweight);

	free(edge_list);
	free(vertex_list);
	free(weight_list);
	free(output_MST_list);
	getchar();
}

