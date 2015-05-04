# cuda_mst
mst alg by cuda running on GPU
dir consrtuct:

./
  MST_VS2012/
  
    MST_64.sdf
    
    MST_64.sln
    
    MST_64.v11.suo
    
    MST_64.v12.suo
    
  bin/
  
    Debug/
    
      MST_64.exe
      
      TransFile.exe
      
      cmp.exe
      
    Release/
    
      MST_64.exe
      
      TransFile.exe
      
      cmp.exe
      
  conf/
  
    ......
    
  src/
    Kernels.cu   
    
    RecMST.cu
    
    RecMST.cuh
    
    TransFile.cpp
    
    cmp.cpp
    
    main.cpp
    
    
    
    We implement an mst alg on GPU by cuda and We compare it with boost mst(minimal spanning tree) alg  
    about data we test the performance of our mst alg by road network of America  
    the url:http://www.dis.uniroma1.it/challenge9/download.shtml.
    
    about input file format :
    vertices num
    list : vertices index each line 
    list : edges and weight of edges each line 
    and we implement a data format program named transfile.exe  also a grogram named cmp.exe compare the results 
    and the reault like that:
  ![](https://raw.githubusercontent.com/yorkhellen/cuda_mst/master/mst.png)  
