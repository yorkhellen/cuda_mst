#ifndef _KERNELS_H_
#define _KERNELS_H_

#define NO_OF_BITS_TO_SPLIT_ON 32
#define MAX_THREADS_PER_BLOCK 512	
#define INF 100000000



////////////////////////////////////////////////////////////////////////////////
//Copy the temporary arrays to the actual arrays, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void ClearSingleEdge(unsigned int *d_edge, 
							unsigned int *d_old_uIDs, 
							unsigned int *d_vertex, 
							unsigned int no_vertex_size,
							unsigned int no_edge_size)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_edge_size)
	{
		unsigned int u = d_old_uIDs[tid];
		unsigned int v = d_edge[tid];
		unsigned int start = d_vertex[v];
		unsigned int end;
		if(v<no_vertex_size-1)
			end = d_vertex[v+1];
		else
			end = no_edge_size;
		int tem = -1;
		for(unsigned int i = start; i < end; i++)
		{
			if(d_edge[i]==u)
				tem=1;//找到对应边
		}
		if(tem==-1)
			d_edge[tid]=INF;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Helper Kernel, Clears a single array
////////////////////////////////////////////////////////////////////////////////
__global__ void ClearArray(unsigned int *d_array, unsigned int size) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<size)
		d_array[tid]=0;
}


////////////////////////////////////////////////////////////////////////////////
// Helper Kernel, Clears multiple arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void ClearArrays(unsigned int *d_edge, unsigned int *d_weight, int *d_pick_array, unsigned int *d_edge_mapping_copy, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
	{
		d_edge[tid]=0;
		d_weight[tid]=0;
		d_pick_array[tid]=0;
	}

}

////////////////////////////////////////////////////////////////////////////////
// Make the flag for Input to the segmented min scan, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlag(unsigned int *d_edge_flag, unsigned int *d_vertex, unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned int pointingvertex = d_vertex[tid];
		d_edge_flag[pointingvertex]=1;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Make the Successor array, Runs for Vertex Length
//d_successor:返回,存放找到的最小权值的边的v点的索引
//d_vertex_flag: 返回,存放找到的最小权值边的索引
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeSucessorArray(unsigned int *d_successor, unsigned int *d_vertex_flag, unsigned int *d_vertex, unsigned int *d_weight, unsigned int *d_edge, unsigned int no_of_vertices, unsigned int no_of_edges) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned int start = d_vertex[tid];
		unsigned int end;
		if(tid<no_of_vertices-1)
			end = d_vertex[tid+1];
		else
			end = no_of_edges;

		unsigned int min_weight = d_weight[start];
		unsigned int minId = start;
//		bool uInVNeignbor;
//		unsigned int vId;
		for(unsigned int i = start; i < end; i++)
		{
			////判断u点是否在v的邻接点集中，否则是有向图，会在合并顶点时，形成不可拆分的环。
			//uInVNeignbor=false;
			//vId = d_edge[i];
			//unsigned int j_start = d_vertex[vId];
			//unsigned int j_end;
			//if(vId<no_of_vertices-1)
			//	j_end = d_vertex[vId+1];
			//else
			//	j_end = no_of_edges;
			//for(unsigned int j=j_start;j<j_end;j++)
			//{
			//	if(tid==j)
			//		uInVNeignbor==true;
			//}
			//找到最小权值的边
			if(d_edge[i]==INF)
			{
				;
			}
			else if(d_weight[i] < min_weight)
			{
				min_weight = d_weight[i];
				minId = i;
			}
			else if(d_weight[i]== min_weight&&d_edge[i]<d_edge[minId])
			{
				min_weight = d_weight[i];
				minId = i;
			}
		}
		d_vertex_flag[tid]=minId;
		d_successor[tid] = d_edge[minId];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Remove Cycles Using Successor array, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void RemoveCycles(unsigned int *d_successor, unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned int succ = d_successor[tid];
		unsigned int nextsucc = d_successor[succ];
		if(tid == nextsucc)//Found a Cycle
		{
			//Give the minimum one its own value, breaking the cycle and setting the Representative Vertices
			if(tid < succ)
				d_successor[tid]=tid;//如果该点索引值比后继索引值小，则设改点的后继为自己
			else
				d_successor[succ]=succ;//如果该点索引值比后继索引值大，则设改点后继的后继为它自己，这个可能不需要，因为小的那个点也会执行该操作
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// Mark Selected Edges in the Output MST array, Runs for Edge Length
//	d_output_MST: 输出值, 边表中属于MST的边置为1
// d_vertex_flag: 输入值,表示某点权值最小的边的索引值 
//	d_successor: 输入值,表示某个顶点的后继
//	d_edge_mapping: 输入值,表示边的索引原始索引值的映射
//	no_of_vertices: 输入值, 顶点个数
////////////////////////////////////////////////////////////////////////////////
__global__ void MarkOutputEdges(bool *d_output_MST,
								unsigned int *d_vertex_flag, 
								unsigned int *d_successor, 
								unsigned int *d_edge_mapping, 
								unsigned int no_of_vertices) 
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		if(tid!=d_successor[tid])			
			d_output_MST[d_edge_mapping[d_vertex_flag[tid]]]=1;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
// Copy from Successor to resolve read after write inconsistancies, Runs for Vertex Length
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void SuccToCopy(unsigned int *d_successor, unsigned int *d_successor_copy, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
		d_successor_copy[tid] = d_successor[tid];
}

////////////////////////////////////////////////////////////////////////////////
// Propagate Representative IDs by setting S(u)=S(S(u)), Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void PropagateRepresentativeID(unsigned int *d_successor, unsigned int *d_successor_copy, bool *d_succchange, unsigned int *temIdx, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned int succ = d_successor[tid];
		unsigned int newsucc = d_successor[succ];
		if(succ!=newsucc) //Execution goes on
		{
			d_successor_copy[tid] = newsucc; //cannot have input and output in the same array!!!!!可不可以加个同步操作，去掉多次拷贝？？？？
			*d_succchange=true;
			*temIdx=tid;
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////
// Copy to Successor to resolve read after write inconsistancies, Runs for Vertex Length
///////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopyToSucc(unsigned int *d_successor, unsigned int *d_successor_copy, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
		d_successor[tid] = d_successor_copy[tid];
}


////////////////////////////////////////////////////////////////////////////////
// Append Vertex IDs with Representative vertex IDs, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void AppendVertexIDsForSort(unsigned long long int *d_vertex_sort, unsigned int *d_successor, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned long long int val;
		val = d_successor[tid];
		val = val<<NO_OF_BITS_TO_SPLIT_ON;
		val |= tid;
		d_vertex_sort[tid]=val;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Mark New SupervertexID per vertex, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeSuperVertexIDPerVertex(unsigned int *d_new_supervertexIDs, 
										   unsigned long long int *d_vertex_sort, 
										   unsigned int *d_vertex_flag,
										   unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		unsigned long long int mask = pow(2.0, NO_OF_BITS_TO_SPLIT_ON);
		mask = mask-1;
		unsigned long long int vertexid = d_vertex_sort[tid]&mask;
		d_vertex_flag[vertexid] = d_new_supervertexIDs[tid];
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copy New SupervertexID per vertex, resolving read after write inconsistancies, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CopySuperVertexIDPerVertex(unsigned int *d_new_supervertexIDs, unsigned int *d_vertex_flag, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
		d_new_supervertexIDs[tid] = d_vertex_flag[tid];
}


////////////////////////////////////////////////////////////////////////////////
// Make flag for Scan, assigning new ids to supervertices, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForScan(unsigned int *d_vertex_flag, unsigned long long int *d_vertex_sort,unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		if(tid>0)
		{
			//unsigned long long int mask = pow(2.0,NO_OF_BITS_TO_SPLIT_ON)-1;
			unsigned long long int val = d_vertex_sort[tid-1];
			unsigned long long int supervertexid_prev  = val>>NO_OF_BITS_TO_SPLIT_ON;
			val = d_vertex_sort[tid];
			unsigned long long int supervertexid  = val>>NO_OF_BITS_TO_SPLIT_ON;
			if(supervertexid_prev!=supervertexid)
				d_vertex_flag[tid]=1;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Make flag to assign old vertex ids, Runs for Vertex Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForUIds(unsigned int *d_edge_flag, unsigned int *d_vertex, unsigned int no_of_vertices)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_vertices)
	{
		if(tid>0)
		{
			int pointingvertex = d_vertex[tid];
			d_edge_flag[pointingvertex]=1;
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// Remove self edges based on new supervertex ids, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void RemoveSelfEdges(unsigned int *d_edge, 
								unsigned int *d_old_uIDs, 
								unsigned int *d_new_supervertexIDs, 
								unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
	{
		unsigned int uid = d_old_uIDs[tid];
		unsigned int vid = d_edge[tid];
		if(vid==INF||uid==INF)
			d_edge[tid]=INF; //Nullify the edge if both vertices have same supervertex id
		else
		{
			unsigned int usuperid = d_new_supervertexIDs[uid];
			unsigned int vsuperid = d_new_supervertexIDs[vid];
			if(usuperid == vsuperid)
				d_edge[tid]=INF; //Nullify the edge if both vertices have same supervertex id
		}
	}
}



////////////////////////////////////////////////////////////////////////////////
// Append U and index for sorting the edge list, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void AppendForNoDuplicateEdgeRemoval(unsigned long long int *d_appended_uindex, 
												unsigned int *d_edge, 
												unsigned int *d_old_uIDs, 
												unsigned int *d_weight, 
												unsigned int *d_new_supervertexIDs, 
												unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
	{
		unsigned long long int val;
		unsigned int u,v,superuid=INF;
		u = d_old_uIDs[tid];
		v = d_edge[tid];
		if(u!=INF && v!=INF)
		{
			superuid = d_new_supervertexIDs[u];
		}
		val = superuid;
		val = val<<NO_OF_BITS_TO_SPLIT_ON;
		val |= tid;
		d_appended_uindex[tid]=val;
/*		unsigned long long int val;

		val = tid;
		val = val<<NO_OF_BITS_TO_SPLIT_ON;
		val |= tid;
		d_appended_uindex[tid]=val;*/
	}
}

////////////////////////////////////////////////////////////////////////////////
// Mark the starting edge for each U for making edge list, Runs for Edge Length
////////////////////////////////////////////////////////////////////////////////
__global__ void MarkEdgesU(unsigned int *d_edge_flag, 
						   unsigned long long int *d_appended_uindex, 
						   unsigned int *d_size, 
						   unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
	{
		if(tid>0)
		{
			unsigned long long int test1 = d_appended_uindex[tid]>>NO_OF_BITS_TO_SPLIT_ON;
			unsigned long long int test2 = d_appended_uindex[tid-1]>>NO_OF_BITS_TO_SPLIT_ON;
			if(test1>test2)
				d_edge_flag[tid]=1;
			if(test1 == INF && test2 != INF)
				*d_size=tid; //去掉内部边后剩余的边数
		}
		else
			d_edge_flag[tid]=1;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////
// Compact the edgelist and weight list, keep a mapping for each edge, Runs for d_size Length
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void CompactEdgeListNoDuplicateRemoval(unsigned int *d_edge, 
												  unsigned int *d_weight, 
												  unsigned int *d_new_supervertexIDs, 
												  unsigned int *d_edge_mapping, 
												  unsigned int *d_segmented_min_scan_input, 
												  unsigned int *d_segmented_min_scan_ouput, 
												  unsigned int* d_edge_mapping_copy,  
												  unsigned int *d_edge_flag, 
												  unsigned long long int *d_appended_uindex, 
												  int *d_pick_array, 
												  unsigned int *d_size, 
												  unsigned int *d_edge_list_size, 
												  unsigned int *d_vertex_list_size)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<*d_size)
	{
		unsigned long long int val = d_appended_uindex[tid]; 
		unsigned long long int mask = pow(2.0, NO_OF_BITS_TO_SPLIT_ON);
		mask = mask-1;
		unsigned long long int index = val&mask;//边的索引值
		unsigned long long int u = val >> NO_OF_BITS_TO_SPLIT_ON;
		unsigned int v = d_edge[index];
		if(u!=INF && v!=INF)
		{
			//Copy the edge_mapping into a temporary array, used to resolve read after write inconsistancies
			d_edge_mapping_copy[tid] = d_edge_mapping[index]; //keep a mapping from old edge-list to new one
			d_pick_array[tid]=u; // reusing this to store u's
			d_segmented_min_scan_ouput[tid]= d_weight[index]; //resuing d_segmented_min_scan_output to store weights
			d_segmented_min_scan_input[tid] = d_new_supervertexIDs[v]; //resuing d_segmented_scan_input to store v ids
			//Set the new vertex list and edge list sizes
			if(tid==*d_size-1)
			{
				*d_edge_list_size=(tid+1);
				*d_vertex_list_size=(u+1);
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//Copy the temporary arrays to the actual arrays, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void CopyEdgeMap(unsigned int *d_edge_mapping, 
							unsigned int *d_edge_mapping_copy, 
							unsigned int *d_edge, 
							unsigned int *d_weight, 
							unsigned int *d_segmented_min_scan_input, 
							unsigned int *d_segmented_min_scan_output, 
							unsigned int *d_size)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<*d_size)
	{
		d_edge_mapping[tid] = d_edge_mapping_copy[tid]; 
		d_edge[tid] = d_segmented_min_scan_input[tid]; 
		d_weight[tid] = d_segmented_min_scan_output[tid]; 
	}
}

////////////////////////////////////////////////////////////////////////////////
//Make Flag for Vertex List Compaction, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeFlagForVertexList(int *d_pick_array, unsigned int *d_edge_flag, unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
	{
		if(tid>0)
		{
			if(d_pick_array[tid]>d_pick_array[tid-1])
				d_edge_flag[tid]=1;
		}
		else
			d_edge_flag[tid]=1;
	}
}

////////////////////////////////////////////////////////////////////////////////
//Vertex List Compaction, Runs for Edge length
////////////////////////////////////////////////////////////////////////////////
__global__ void MakeVertexList(unsigned int *d_vertex, int *d_pick_array, unsigned int *d_edge_flag,unsigned int no_of_edges)
{
	unsigned int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(tid<no_of_edges)
	{
		if(d_edge_flag[tid]==1)
		{
			unsigned int writepos=d_pick_array[tid]; //get the u value
			d_vertex[writepos]=tid; //write the index to the u'th value in the array to create the vertex list
		}
	}
}


#endif // #ifndef _KERNELS_H_
