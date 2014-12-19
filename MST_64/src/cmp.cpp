//  [9/24/2014 York]
/*<summary>
  for test the result of gpu and cpu 
  </summary>
*/
#include <stdio.h>
#include <vector>
using namespace std;

struct edge
{
	int u;
	int v;  
	int w;
} ;
/* [10/9/2014 York]
 * describe: read results and compare the results
 */
int  cmp(char *src, char *des)
{
    //read file 
	vector<edge> srcarray;
	vector<edge> desarray;
	FILE *fp,*fl;
	fp=fopen(src,"r");
	fl=fopen(des,"r");
	if(fp ==NULL ||fl == NULL)
		return 0;
	printf("Reading result...\n");
	edge e ;
	while (fscanf(fp,"%d %d %d",&(e.u),&(e.v),&(e.w)) !=EOF)
		srcarray.push_back(e);
	while (fscanf(fl,"%d %d %d",&(e.u),&(e.v),&(e.w)) !=EOF)
		desarray.push_back(e);
	fclose(fp);
	fclose(fl);
	// if gpu mst length != cpu mst length
	 if(srcarray.size() != desarray.size())
		 return 0;
	int len = srcarray.size();
	double w_a,w_b;
	w_b=0;
	w_a=0;
	for(int it_i = 0 ; it_i <len ; it_i ++)
	{
		w_b+=desarray[it_i].w;
		w_a+=srcarray[it_i].w;
	}
//  compare the total weight of two results
	  if (w_a ==w_b)
	  {
		  return 1;
	  }
		 return 0;;
}
	
int main(int argv ,char ** argc)
{
  if(argv <3)
	  {
		  printf("please input the result of gpu mst and the corrent result\n");
		  exit(1);
	  }
  if (cmp(argc[1],argc[2])==1)
	  printf("right");
  else
	  printf("error");
  
}