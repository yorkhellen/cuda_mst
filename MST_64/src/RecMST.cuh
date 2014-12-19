#ifndef _RecMST_H_
#define _RecMST_H_

class MST
{

public:
	////////////////////////////////////////////////
	// Variables
	////////////////////////////////////////////////
	unsigned int no_of_vertices;		//vertex sizes
	unsigned int no_of_edges;			//edge sizes
	unsigned int *d_edge, *d_vertex, *d_weight;			//Graph held in these variables at the device end
	unsigned int *d_segmented_min_scan_input;			//Input to the Segmented Min Scan, appended array
	unsigned int *d_segmented_min_scan_output;			//Output of the Segmented Min Scan
	unsigned int *d_edge_flag;					//Flag for the segmented min scan
	unsigned int *d_vertex_flag;					//Flag for the scan input for supervertex ID generation
	bool *d_output_MST;					//Final output, marks 1 for selected edges in MST, 0 otherwise
	int *d_pick_array;						//PickArray for each edge
	unsigned int *d_successor;					//Successor Array
	unsigned int *d_successor_copy;					//Copy of Successor Array
	bool *d_succchange;						//Variable to check for execution while propagating representative vertex IDs
	unsigned long long int *d_vertex_sort;				//Input to the sort function
	unsigned int *d_new_supervertexIDs;				//new supervertex ids after scanning older IDs
	unsigned int *d_old_uIDs;					//old ids, stored per edge, needed to remove self edges
	unsigned long long int *d_appended_uindex;			//Appended u,index array for edge sorting
	unsigned int *d_size;						//Variable to hold new size of grid
	unsigned int *d_edge_mapping;					//The edge mapping array used across invocations
	unsigned int *d_edge_mapping_copy;				//Copy of the mapping aray
	unsigned int *d_edge_list_size;					//New edge list size
	unsigned int *d_vertex_list_size;				//New vertex list size
	int iteration;						//Counting Iterations

public:
	void Init(unsigned int *h_edge_list, unsigned int *h_vertex_list, unsigned int *h_weight_list, unsigned int no_of_vertices, unsigned int no_of_edges);
	void SetGridThreadLen(unsigned int number, unsigned int *num_of_blocks, unsigned int *num_of_threads_per_block);
	void DoMST(bool *output_MST_list, unsigned int h_no_of_edges);
	void RecMST();
	void FreeMem();
};
#endif