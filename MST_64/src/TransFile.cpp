//  [9/17/2014 zxy]
/*<summary>
 
  </summary>
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

void trans(char *srcfilename,char *desfilename)
{ 
    unsigned int no_of_vertices =0;
    unsigned int no_of_edges=0;

	printf("\nReading Graph...\n");
	FILE *fp;
	fp = fopen(srcfilename,"r");
	if(fp == NULL) 
	{	
		printf("Can not open srcfile... ");
		return ;
	}
	char Strline[256];
	for (unsigned int sc = 0 ;sc<4;sc++)  fgets(Strline,256,fp);
	char c;
	fscanf(fp,"%c %s %d %d",&c,Strline,&no_of_vertices,&no_of_edges);
	//store the data from srcfile
	unsigned int *file_graph =  (unsigned int*)malloc(sizeof(unsigned int)*(no_of_edges+no_of_vertices+2)*2);
	//set the number of vertices and the number of edges
	*(file_graph+0)= no_of_vertices;
	*(file_graph+((no_of_vertices)+1)*2)=no_of_edges;

	fgets(Strline,256,fp);
	fgets(Strline,256,fp);
	fgets(Strline,256,fp);

	// set the number of adjacent edges of each vertex 0
	for (unsigned int j = 1;j<= no_of_vertices ;j++)
		*(file_graph +j*2+1) =0;

	// statistics the number of  adjacent edges of each vertex
	int u,v,w;
	for(unsigned int number_of_adjacent_edge = 0; number_of_adjacent_edge <no_of_edges;number_of_adjacent_edge++)
	{
		fscanf(fp,"%c %d %d %d\n",&c,&u, &v,&w);
		*(file_graph+u*2+1)=*(file_graph+u*2+1)+1;
	}
	/* set the begin index in the edges array for each vertex */
	// set the first vertex
	*(file_graph+2)=0;
	for(unsigned int index = 2 ;index<=no_of_vertices;index++)
		*(file_graph+index*2) =*(file_graph+(index-1)*2) +*(file_graph+(index-1)*2+1);
	// set each edges array for each vertex
	unsigned int *index_edges =(unsigned int*)malloc(sizeof(unsigned int)*(no_of_vertices+1));
	for ( unsigned int index=1;index<=no_of_vertices;index++)
		*(index_edges+index)=0;
	fclose(fp);

	fp = fopen(srcfilename,"r");
	if(fp == NULL) return ;
	for (unsigned int sc = 0 ;sc<7;sc++)  fgets(Strline,256,fp);
	for(unsigned int number_of_adjacent_edge = 0 ; number_of_adjacent_edge <no_of_edges;number_of_adjacent_edge++)
	{
		fscanf(fp,"%c %d %d %d\n",&c,&u, &v,&w) ;
		//set the adjacent vertex
		*(file_graph+(2+no_of_vertices+*(file_graph+u*2)+*(index_edges+u))*2)=v;
		//set the weight of edge
		*(file_graph+(2+no_of_vertices+*(file_graph+u*2)+*(index_edges+u))*2+1)=w;
		//increase the index of edges for each vertex
		*(index_edges+u)=*(index_edges+u)+1;
	}
	fclose(fp);

	//writing graph
	printf("\nWriting file Graph......\n");
     fp = fopen(desfilename,"w+");
	  if(fp == NULL) return ;
	  fprintf(fp,"%d\n",no_of_vertices);
	  int start,index;
	for (unsigned int i = 0 ; i < no_of_vertices ; i++ )
	{
		start =*(file_graph+(i+1)*2);
		index =*(file_graph+(i+1)*2+1);
		fprintf(fp,"%d %d\n",start,index);
	}
	fprintf(fp,"\n");
	fprintf(fp,"%d\n",0);
	fprintf(fp,"\n");
	fprintf(fp,"%d\n",no_of_edges);
	for( unsigned int i = 0 ; i < no_of_edges ; i++ )
		fprintf(fp,"%d %d\n",*(file_graph+(2+no_of_vertices+i)*2)-1,*(file_graph+(2+no_of_vertices+i)*2+1));
	fclose(fp);
	free(index_edges);
	free(file_graph);
	printf("trans  successfully.\nNumber of Vertices: %d\nNumber of Edges: %d\n", no_of_vertices, no_of_edges);
}

int main(int argc,char ** argv)
{
	if(argc <3)
	{
		printf("error args:the first arg is srcfile and the second arg is desfile\n");
	   exit(1);
	}
	// argv[1] :原数据文件路径
    // argv[2] :目标文件路径
	trans(argv[1],argv[2]);
	
}