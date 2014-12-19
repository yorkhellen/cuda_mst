#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>

#include "RecMST.cuh"
#include "Kernels.cu"
#include <windows.h>


////////////////////////////////////////////////
// Allocate and Initialize Arrays
////////////////////////////////////////////////
void MST::Init(unsigned int *h_edge_list, unsigned int *h_vertex_list, unsigned int *h_weight_list, unsigned int h_no_of_vertices, unsigned int h_no_of_edges)
{
 
	iteration=0;
	no_of_vertices=h_no_of_vertices;
	no_of_edges=h_no_of_edges;
	//Couning total space used on device
	int totaldevicemem=0;

	//Allocate main graph variables
	( cudaMalloc( (void**) &d_edge, sizeof(unsigned int)*no_of_edges));
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;
	( cudaMalloc( (void**) &d_vertex, sizeof(unsigned int)*no_of_vertices));
	totaldevicemem+=sizeof(unsigned int)*no_of_vertices;
	( cudaMalloc( (void**) &d_weight, sizeof(unsigned int)*no_of_edges));
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;

	//Copy the Graph to Device
	( cudaMemcpy( d_edge, h_edge_list, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice));
	( cudaMemcpy( d_vertex, h_vertex_list, sizeof(unsigned int)*no_of_vertices, cudaMemcpyHostToDevice));
	( cudaMemcpy( d_weight, h_weight_list, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice));
	printf("Graph Copied to Device.\n");

	//Allocate memory for other arrays
	(cudaMalloc( (void**) &d_segmented_min_scan_input, sizeof(unsigned int)*no_of_edges)); 
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;
	(cudaMalloc( (void**) &d_segmented_min_scan_output, sizeof(unsigned int)*no_of_edges));
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;
	(cudaMalloc( (void**) &d_edge_flag, sizeof(unsigned int)*no_of_edges));
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;
	(cudaMalloc( (void**) &d_pick_array, sizeof(int)*no_of_edges));
	totaldevicemem+=sizeof(int)*no_of_edges;
	(cudaMalloc( (void**) &d_successor,sizeof(unsigned int)*no_of_vertices));
	totaldevicemem+=sizeof(int)*no_of_vertices;
	(cudaMalloc( (void**) &d_successor_copy,sizeof(int)*no_of_vertices));
	totaldevicemem+=sizeof(int)*no_of_vertices;
	(cudaMalloc( (void**) &d_output_MST, sizeof(bool)*no_of_edges));
	totaldevicemem+=sizeof(int)*no_of_edges;

	//Clear Output MST array
	unsigned int *h_test=(unsigned int*)malloc(sizeof(unsigned int)*no_of_edges);
	for(unsigned int i=0;i<no_of_edges;i++)
		h_test[i]=0;//���Ըĳ�memset��
	( cudaMemcpy( d_output_MST, h_test, sizeof(bool)*no_of_edges, cudaMemcpyHostToDevice));

	(cudaMalloc( (void**) &d_succchange, sizeof(bool)));
	totaldevicemem+=sizeof(bool);
	(cudaMalloc( (void**) &d_vertex_sort, sizeof(unsigned long long int)*no_of_vertices));
	totaldevicemem+=sizeof(unsigned long long int)*no_of_vertices;
	(cudaMalloc( (void**) &d_vertex_flag, sizeof(unsigned int)*no_of_vertices));
	totaldevicemem+=sizeof(unsigned int)*no_of_vertices;
	(cudaMalloc( (void**) &d_new_supervertexIDs, sizeof(unsigned int)*no_of_vertices));
	totaldevicemem+=sizeof(unsigned int)*no_of_vertices;
	(cudaMalloc( (void**) &d_old_uIDs, sizeof(unsigned int)*no_of_edges));
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;
	(cudaMalloc( (void**) &d_appended_uindex, sizeof(unsigned long long int)*no_of_edges));
	totaldevicemem+=sizeof(unsigned long long int)*no_of_edges;
	(cudaMalloc( (void**) &d_size, sizeof(unsigned int)));
	totaldevicemem+=sizeof(unsigned int);
	(cudaMalloc( (void**) &d_edge_mapping, sizeof(unsigned int)*no_of_edges)); 
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;
	(cudaMalloc( (void**) &d_edge_mapping_copy, sizeof(unsigned int)*no_of_edges)); 
	totaldevicemem+=sizeof(unsigned int)*no_of_edges;

	//Initiaize the d_edge_mapping array
	for(unsigned int i=0;i<no_of_edges;i++)
		h_test[i]=i;
	( cudaMemcpy( d_edge_mapping, h_test, sizeof(unsigned int)*no_of_edges, cudaMemcpyHostToDevice));
	free(h_test);

	(cudaMalloc( (void**) &d_edge_list_size, sizeof(unsigned int)));
	totaldevicemem+=sizeof(unsigned int);
	(cudaMalloc( (void**) &d_vertex_list_size, sizeof(unsigned int)));
	totaldevicemem+=sizeof(unsigned int);

	//��������
	unsigned int num_of_blocks, num_of_threads_per_block;
	SetGridThreadLen(no_of_edges, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_edgelen(num_of_blocks, 1, 1);
	dim3 threads_edgelen(num_of_threads_per_block, 1, 1);
	
	SetGridThreadLen(no_of_vertices, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_vertexlen(num_of_blocks, 1, 1);
	dim3 threads_vertexlen(num_of_threads_per_block, 1, 1);
	



	//����inclusive_scan����ÿ���ߵ���ʼ�ڵ�u������ֵ��������������d_old_uIDs�С�
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );
	MakeFlagForUIds<<< grid_vertexlen, threads_vertexlen, 0>>>(d_edge_flag, d_vertex,no_of_vertices);
	thrust::inclusive_scan(thrust::device_ptr<unsigned int>(d_edge_flag), thrust::device_ptr<unsigned int>(d_edge_flag+no_of_edges), thrust::device_ptr<unsigned int>(d_old_uIDs));
	ClearSingleEdge<<< grid_edgelen, threads_vertexlen, 0>>>(d_edge,d_old_uIDs, d_vertex, no_of_vertices,no_of_edges);
	
	printf("Total Device Memory Used: %.3f MB\n",totaldevicemem/(1024.0*1024.0));
}


////////////////////////////////////////////////
// Helper function to set the grid sizes
////////////////////////////////////////////////
void MST::SetGridThreadLen(unsigned int number, unsigned int *num_of_blocks, unsigned int *num_of_threads_per_block)
{
	*num_of_blocks = 1;
	*num_of_threads_per_block = number;

	//Make execution Parameters according to the input size
	//Distribute threads across multiple Blocks if necessary
	if(number>MAX_THREADS_PER_BLOCK)
	{
		*num_of_blocks = (unsigned int)ceil(number/(float)MAX_THREADS_PER_BLOCK); 
		*num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
}


////////////////////////////////////////////////
// Perform Our Recursive MST Algorithm
////////////////////////////////////////////////
void MST::RecMST()
{

	//Make both grids needed for execution, no_of_vertices and no_of_edges length sizes
	unsigned int num_of_blocks, num_of_threads_per_block;
	SetGridThreadLen(no_of_edges, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_edgelen(num_of_blocks, 1, 1);
	dim3 threads_edgelen(num_of_threads_per_block, 1, 1);
	
	SetGridThreadLen(no_of_vertices, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_vertexlen(num_of_blocks, 1, 1);
	dim3 threads_vertexlen(num_of_threads_per_block, 1, 1);

	//����inclusive_scan����ÿ���ߵ���ʼ�ڵ�u������ֵ��������������d_old_uIDs�С�
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );
	MakeFlagForUIds<<< grid_vertexlen, threads_vertexlen, 0>>>(d_edge_flag, d_vertex,no_of_vertices);
	thrust::inclusive_scan(thrust::device_ptr<unsigned int>(d_edge_flag), thrust::device_ptr<unsigned int>(d_edge_flag+no_of_edges), thrust::device_ptr<unsigned int>(d_old_uIDs));
	 
	//Ϊÿ������ҵ�Ȩֵ��С�ıߣ��Ѹñߵĳ���v����ں������d_successor�У��ñߵ������������ʱ����d_vertex_flag�С�
	MakeSucessorArray<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_vertex_flag, d_vertex, d_weight, d_edge, no_of_vertices, no_of_edges);

	//�������е�ѭ�����󣬼����ĳ���ڵ�ĺ�̵ĺ�����Լ����Լ�������ֵ�Ⱥ�̵�С���򽫸ý��ĺ���޸�Ϊ�Լ�������ֵ��
	//��������ͼ������ͼ����������������ֺ��ѭ�����󡣵����������ͼ�����Ծɿ��ܳ���ѭ�����󣬱������⴦��
	RemoveCycles<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor,no_of_vertices);

//	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>((unsigned int*)d_pick_array, no_of_edges);

	//��ǳ�ÿ���ڵ�Ȩֵ��С�ıߣ���������d_output_MST�С�
	MarkOutputEdges<<< grid_edgelen, threads_vertexlen, 0>>>(d_output_MST,d_vertex_flag, d_successor, d_edge_mapping,no_of_vertices);
    unsigned int* d_temIdx;
	unsigned int h_temIdx;
	(cudaMalloc( (void**) &d_temIdx, sizeof(unsigned int)));

	//��ÿ�����ĺ���ú�̵ĺ�̴��棬ֱ����̵��ں�̵ĺ�����Լ�Ϊֹ�������˱߽��
	bool succchange;
	do 
	{
		succchange=false; //if no thread changes this value, the loop stops
		h_temIdx=-1;
		( cudaMemcpy( d_succchange, &succchange, sizeof(bool), cudaMemcpyHostToDevice));
		( cudaMemcpy( d_temIdx, &h_temIdx, sizeof(unsigned int), cudaMemcpyHostToDevice));
		SuccToCopy<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, no_of_vertices);
		PropagateRepresentativeID<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, d_succchange,d_temIdx, no_of_vertices);
		CopyToSucc<<< grid_vertexlen, threads_vertexlen, 0>>>(d_successor, d_successor_copy, no_of_vertices);
		( cudaMemcpy( &succchange, d_succchange, sizeof(bool), cudaMemcpyDeviceToHost));
		( cudaMemcpy( &h_temIdx, d_temIdx, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	} 
	while(succchange);
	 
	//��ÿ�����ĺ�̺��Լ�������ƴ�ӳ�64λ�����������򣬱���������d_vertex_sort�С�
	AppendVertexIDsForSort<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_sort, d_successor,no_of_vertices);
	thrust::device_ptr<unsigned long long int> dev_ptr_vertex(d_vertex_sort);
	thrust::sort(dev_ptr_vertex, dev_ptr_vertex + no_of_vertices);

	//��ÿ�������ͬ�Ľ�����ͬ����һ����supervertex�������������d_new_supervertexIDs
	ClearArray<<< grid_vertexlen, threads_vertexlen, 0>>>( d_vertex_flag, no_of_vertices);
	MakeFlagForScan<<< grid_vertexlen, threads_vertexlen, 0>>>(d_vertex_flag, d_vertex_sort, no_of_vertices);
	thrust::inclusive_scan(thrust::device_ptr<unsigned int>(d_vertex_flag), thrust::device_ptr<unsigned int>(d_vertex_flag+no_of_vertices), thrust::device_ptr<unsigned int>(d_new_supervertexIDs));

	//�����¾ɽ�������Ķ��ձ�����ʱ���������d_vertex_flag��
	MakeSuperVertexIDPerVertex<<< grid_vertexlen, threads_vertexlen, 0>>>(d_new_supervertexIDs, d_vertex_sort, d_vertex_flag, no_of_vertices);
	//���¾ɽ�������Ķ��ձ���������d_new_supervertexIDs��
	CopySuperVertexIDPerVertex<<< grid_vertexlen, threads_vertexlen, 0>>>(d_new_supervertexIDs, d_vertex_flag, no_of_vertices);

	//�����ʼ��u����ֹ��v��ָ��ͬһ��supervertex�ıߣ�����d_edge�иñ�ָ��Ľڵ���ΪINF
	RemoveSelfEdges<<< grid_edgelen, threads_edgelen, 0>>>(d_edge, d_old_uIDs, d_new_supervertexIDs, no_of_edges);
/*	  */
	//��ÿ���ߵ���ʼ��u��Ӧ��supervertexֵ�ͱߵ�����ֵƴ�ӳ�64λֵ��������
	AppendForNoDuplicateEdgeRemoval<<< grid_edgelen, threads_edgelen, 0>>>(d_appended_uindex, d_edge, d_old_uIDs, d_weight,d_new_supervertexIDs, no_of_edges);
	thrust::device_ptr<unsigned long long int> dev_ptr(d_appended_uindex);
	thrust::sort(dev_ptr, dev_ptr + no_of_edges);
	//������d_appended_uindex�г���u��ͬ��λ�ñ��Ϊ1����һ��λ��Ҳ���Ϊ1�������d_edge_flag�У�ͬʱ���ʣ��ı��������d_size
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges );
	unsigned int dsize=no_of_edges+1; //just to be sure
	( cudaMemcpy( d_size, &dsize, sizeof(unsigned int), cudaMemcpyHostToDevice));
	MarkEdgesU<<< grid_edgelen, threads_edgelen, 0>>>(d_edge_flag, d_appended_uindex, d_size, no_of_edges);

	//��ʼ�� d_segmented_min_scan_input as edge_copy, d_segmented_min_scan_output as weight_copy, pick_array for new u's per edge
	ClearArrays<<< grid_edgelen, threads_edgelen, 0>>>( d_segmented_min_scan_input, d_segmented_min_scan_output , d_pick_array, d_edge_mapping_copy, no_of_edges );

	unsigned int zero=0;
	( cudaMemcpy( d_edge_list_size, &zero, sizeof( unsigned int), cudaMemcpyHostToDevice));
	( cudaMemcpy( d_vertex_list_size, &zero, sizeof( unsigned int), cudaMemcpyHostToDevice));

	//�����������в���
	cudaMemcpy( &dsize, d_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	SetGridThreadLen(dsize, &num_of_blocks, &num_of_threads_per_block);
	dim3 grid_validsizelen(num_of_blocks, 1, 1);
	dim3 threads_validsizelen(num_of_threads_per_block, 1, 1);
	//��û�п��������߿�������ͬ��supervertex�����ڵ��㷨�ǵȵ���һ�ִ���ʱ����ȥ���ظ���һ�����᲻��������⣿
	//Reusing d_pick_array for storing the new u ids
	CompactEdgeListNoDuplicateRemoval<<< grid_validsizelen, threads_validsizelen, 0>>>(d_edge, d_weight, d_new_supervertexIDs, d_edge_mapping, d_segmented_min_scan_input, d_segmented_min_scan_output, d_edge_mapping_copy, d_edge_flag, d_appended_uindex, d_pick_array, d_size, d_edge_list_size, d_vertex_list_size);

	//Copy the arrays back to actual arrays, this is used to resolve read after write inconsistancies
	CopyEdgeMap<<<grid_validsizelen, threads_validsizelen, 0 >>>(d_edge_mapping, d_edge_mapping_copy, d_edge, d_weight, d_segmented_min_scan_input, d_segmented_min_scan_output, d_size);


	//Make the vertex list
	//���¶�������
	ClearArray<<< grid_edgelen, threads_edgelen, 0>>>( d_edge_flag, no_of_edges);
	ClearArray<<< grid_vertexlen, threads_vertexlen, 0>>>((unsigned int*)d_vertex, no_of_vertices);
	MakeFlagForVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_pick_array, d_edge_flag, no_of_edges);
	MakeVertexList<<< grid_edgelen, threads_edgelen, 0>>>(d_vertex, d_pick_array, d_edge_flag, no_of_edges);

//	unsigned int h_tem[6];
//	CUDA_SAFE_CALL( cudaMemcpy( h_tem, d_vertex, sizeof(unsigned int)*6, cudaMemcpyDeviceToHost));

	//Copy back the new sizes of vertex list and edge list
	( cudaMemcpy( &no_of_edges, d_edge_list_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));
	( cudaMemcpy( &no_of_vertices, d_vertex_list_size, sizeof(unsigned int), cudaMemcpyDeviceToHost));

	//Do the same again on the newly created graph until a single vertex remains...
}

void MST::DoMST(bool *output_MST_list, unsigned int h_no_of_edges)
{
	printf("\nPerforming MST on Graph, without duplicate edge removal.\n");

    // create cuda event handles
/*    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    checkCudaErrors(cudaDeviceSynchronize());
    float gpu_time = 0.0f;

    // asynchronously issue work to the GPU (all to stream 0)
    sdkStartTimer(&timer);
    cudaEventRecord(start, 0);
	*/
	//Perform Our MST algorihtm
	double  start = GetTickCount() ;
	do
	{
		RecMST();
		iteration++;
	}
	while(no_of_vertices>1);
	double  stop = GetTickCount() ;
	cudaThreadSynchronize();
 /*   cudaEventRecord(stop, 0);
    sdkStopTimer(&timer);
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));*/
	printf("the gpu time is : %f s \n",(stop-start)/1000);
	//printf("\n=================== Time taken To perform MST in %d Iterations :: %3.3f ms ===================\n", iteration, 0);

	//Copy the Final MST array to the CPU memory, a 1 at the index means that edge was selected in the MST, 0 otherwise.
	//It should be noted that each edge has an opposite edge also, out of which only one is selected in this output.
	//So total number of 1s in this array must be equal to no_of_vertices_orig-1.
	( cudaMemcpy( output_MST_list, d_output_MST, sizeof(bool)*h_no_of_edges, cudaMemcpyDeviceToHost));
}



////////////////////////////////////////////////
//Free All memory from Host and Device
////////////////////////////////////////////////
void MST::FreeMem()
{
	(cudaFree(d_edge));
	(cudaFree(d_vertex));
	(cudaFree(d_weight));
	(cudaFree(d_segmented_min_scan_input));
	(cudaFree(d_segmented_min_scan_output));
	(cudaFree(d_edge_flag));
	(cudaFree(d_pick_array));
	(cudaFree(d_successor));
	(cudaFree(d_successor_copy));
	(cudaFree(d_output_MST));
	(cudaFree(d_succchange));
	(cudaFree(d_vertex_sort));
	(cudaFree(d_vertex_flag));
	(cudaFree(d_new_supervertexIDs));
	(cudaFree(d_old_uIDs));
	(cudaFree(d_appended_uindex));
	(cudaFree(d_size));
	(cudaFree(d_edge_mapping));
	(cudaFree(d_edge_mapping_copy));
	(cudaFree(d_edge_list_size));
	(cudaFree(d_vertex_list_size));

}



