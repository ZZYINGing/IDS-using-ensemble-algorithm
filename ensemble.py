import h5py
from collections import Counter
import numpy as  np
import copy
import argparse

#generate cluster matrix
def binary_vectorize(label_array,no_of_clusters):
  binary_vector = []
  for i in range(len(no_of_clusters)):
    for j in range(no_of_clusters[i]):
      temp_arr = (label_array[i] == j)
      binary_vector.append(temp_arr*1)
  return np.array(binary_vector)

#generate cluster matrix(calculate the similarity)
def cluster_similarity_score(cluster1, cluster2, no_of_objects):
  sum_c1 = np.sum(cluster1)
  sum_c2 = np.sum(cluster2)
  #print(cluster1,cluster2)
  c1_intersect_c21 = cluster1[cluster1==cluster2]
  c1_intersect_c2 = sum(c1_intersect_c21)
  cm1 = round((sum_c1*sum_c2)**0.5,3)
  cm = round(c1_intersect_c2/cm1,3)
  #print(c1_intersect_c2,cm,cm1,sum_c1,sum_c2,(no_of_objects*cm-(8)**0.5))
  similarity_score = round(((no_of_objects*cm)-cm1)/round(((no_of_objects-sum_c1)*(no_of_objects-sum_c2))**0.5,3),3)
  return similarity_score

#generate similarity matrix
def similarity_matrix(no_of_techniques,array_of_individual_no_of_clusters,clusters):
  total_cluster_number=sum(array_of_individual_no_of_clusters)
  matrix=[[0]*total_cluster_number for i in range(total_cluster_number)]
  #print(total_cluster_number,matrix)
  for i in range(no_of_techniques):
    num_clust=array_of_individual_no_of_clusters[i]
    for j in range(num_clust):
      start_point=sum(array_of_individual_no_of_clusters[0:i])
      for k in range(total_cluster_number-num_clust):
        comparing_cluster_index=(start_point+num_clust+k)%total_cluster_number
        #print(i,j+start_point,comparing_cluster_index)
        matrix[j+start_point][comparing_cluster_index]=cluster_similarity_score(clusters[j+start_point],clusters[comparing_cluster_index],len(clusters[0]))
  return matrix

#merging step 
def merging(no_of_techniques,array_of_individual_no_of_clusters,matrix,clusters,alpha):
  #need to reduce repeated code
  merged=[]
  new_matrix=[]
  total_cluster_number=sum(array_of_individual_no_of_clusters)
  for i in range(no_of_techniques):
    num_clust=array_of_individual_no_of_clusters[i]
    for j in range(num_clust):
      #print(clusters)
      start_point=sum(array_of_individual_no_of_clusters[0:i])
      subsum=clusters[j+start_point]
      if j+start_point in merged:
        continue
      for k in range(total_cluster_number-num_clust):
        comparing_cluster_index=(start_point+num_clust+k)%total_cluster_number
        #print(i,j+start_point,comparing_cluster_index,matrix[j+start_point][comparing_cluster_index],subsum,clusters[comparing_cluster_index])
        if (comparing_cluster_index in merged):
          continue
        if matrix[j+start_point][comparing_cluster_index]>=alpha:
          temp=copy.deepcopy(clusters[comparing_cluster_index])
          #print(temp)
          merged.append(j+start_point)
          merged.append(comparing_cluster_index)
          subsum=subsum+temp
      new_matrix.append(subsum)
      #print(new_matrix)
  return np.array(new_matrix)

#get uncertainty 
def object_similarity_score(clusters):
  s=0
  print("cluster",clusters)
  for i in clusters:
    s+=i[0]
  
  for i in range(len(clusters)):
    clusters[i]=clusters[i]/s
  similarity_score_cluster=clusters
  print("a",similarity_score_cluster)
  return np.array(similarity_score_cluster)
def final_output(oss,alpha2,index):
  output1=[]
  output2= []
  #np.array()
  for i in range(len(oss)):
    output1.append([])
    output2.append([])
  #print(output1)
  list_obj=[x for x in range(0,len(oss[0]))]
  #print(list_obj)
  for i in range(len(index)):
    if (oss[index[i]][i]>=alpha2):
      output1[index[i]].append(oss[index[i]][i])
      output2[index[i]].append(i)
      list_obj.remove(i)

  return output1,output2,list_obj

def calculate_quality_score(object_cluster):
  s = []
  for i in object_cluster:
    if (len(i)!=0):
        s.append(np.var(np.array(i)))
    else:
      s.append(0)
  return np.array(s)

def updated_quality_score(quality_score,uncertain_score,object_cluster):
  for i in range(len(object_cluster)):
    object_cluster[i].append(uncertain_score)
  new_quality_score=calculate_quality_score(object_cluster)
  return np.argmin(np.abs(quality_score-new_quality_score))
  

  
  
  s_matrix = similarity_matrix(no_of_techniques,no_of_cluster_array,binary_vector)


  while (True):
    new_matrix=merging(no_of_techniques,no_of_cluster_array,s_matrix,binary_vector,alpha1)
    lamda = len(new_matrix)
    if (lamda>=needed_cluster):
      break
    else:
      alpha1 +=delta_alpha


  while len(new_matrix)>needed_cluster:
    new_matrix_2=(new_matrix!=0)*1
    new_s_matrix=similarity_matrix(len(new_matrix_2),[1]*len(new_matrix_2),new_matrix_2)
    alpha1=max(np.amax(new_s_matrix,axis=0))
    if alpha1<alpha_min:
      break
    else:
      new_merged_matrix=merging(len(new_matrix_2),[1]*len(new_matrix_2),new_s_matrix,new_matrix,alpha1)
    if len(new_merged_matrix)<=needed_cluster:
      break
    else:
      new_matrix=new_merged_matrix

  
  oss=object_similarity_score(new_merged_matrix)
  index=np.argmax(oss,axis=0)


  output1,cluster_with_object,uncertain_cluster=final_output(oss,alpha2,index)


  if (len(uncertain_cluster)!=0):
    '''
    #print(cluster_with_object)
    
  else:'''
    q_score = calculate_quality_score(output1)
    for iter in uncertain_cluster:
      indi_cluster = updated_quality_score(q_score,oss[index[iter]][iter],output1)
      output1[indi_cluster].append(oss[index[iter]][iter])
      cluster_with_object[indi_cluster].append(iter)
      q_score = calculate_quality_score(output1)
    #print(cluster_with_object)






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', "--number_clusters", type=int, default=2)
    parser.add_argument("-i", "--vectors", required=True, help="HDF5 file containing the vectors")
    parser.add_argument("-o", "--output", required=True, help="Output HDF5 containing the vectors")
    args = parser.parse_args()
    output_path = args.output
    paths = args.vectors.split(",")
    #print(type(paths),paths)
    
    techniques=len(paths)
    alpha1=0.5
    alpha2=0.4
    alpha_min=0.3
    needed_cluster = args.number_clusters
    delta_alpha = 0.05
    
    arr_clust=[]
    no_clust=[]
    new_merged_matrix=[]
    for path in paths:
        with h5py.File(path, "r") as f:
            vectors = f["vectors"][:]
            ips = f["notes"][:]
            clusters = f["cluster"][:]
        #print("v",vectors,"\nc",clusters,"\nips",ips)
        arr_clust.append(clusters)
        counter = Counter(clusters.tolist())
        no_clust.append(len(counter.keys()))
    #print(no_clust,arr_clust)
        
    k=binary_vectorize(arr_clust,no_clust)
    print(len(k),len(k[0]))
    
    s_matrix = similarity_matrix(techniques,no_clust,k)
    #print(s_matrix)

    new_matrix=merging(techniques,no_clust,s_matrix,k,alpha1)
    #print(new_matrix)
    
    while(True):
        new_matrix=merging(techniques,no_clust,s_matrix,k,alpha1)
        lamda = len(new_matrix)
        if (lamda>=needed_cluster):
            break
        else:
            alpha1 +=delta_alpha
    print(new_matrix,"needed clust",needed_cluster)

    while len(new_matrix)>needed_cluster:
        #print("in")
        new_matrix_2=(new_matrix!=0)*1
        #print(new_matrix_2)
        new_s_matrix=similarity_matrix(len(new_matrix_2),[1]*len(new_matrix_2),new_matrix_2)
        #print(np.array(new_s_matrix))
        alpha1=max(np.amax(new_s_matrix,axis=0))
        if alpha1<alpha_min:
            # print("out")
            break
        else:
            new_merged_matrix=merging(len(new_matrix_2),[1]*len(new_matrix_2),new_s_matrix,new_matrix,alpha1)
        if len(new_merged_matrix)<=needed_cluster:
            #print("hey")
            break
        else:
            #print("hello")
            new_matrix=new_merged_matrix
    print("nmm",new_merged_matrix,len(new_matrix))

    oss=object_similarity_score(new_merged_matrix)
    #oss>0.6
    
    oss.transpose()
    print("hi\n",oss)
    
    index=np.argmax(oss,axis=0)
    output1,cluster_with_object,uncertain_cluster=final_output(oss,alpha2,index)
    updated_quality_score(calculate_quality_score(output1),0.35,output1)
    #put objects into clusters 

    if(len(uncertain_cluster)!=0):
      '''#print(cluster_with_object)
      continue
    else:'''
      q_score = calculate_quality_score(output1)
      for iter in uncertain_cluster:
        indi_cluster = updated_quality_score(q_score,oss[index[iter]][iter],output1)
        output1[indi_cluster].append(oss[index[iter]][iter])
        cluster_with_object[indi_cluster].append(iter)
        q_score = calculate_quality_score(output1)
      #print(cluster_with_object)
      for i in range(len(cluster_with_object)):
          for j in range(len(cluster_with_object[i])):
              #print(j,i,cluster_with_object[i],cluster_with_object[i][j])
              clusters[cluster_with_object[i][j]]=i
      #print(clusters)
      #print(clusters)
    '''
    s_matrix = similarity_matrix(techniques,no_clust,k)


    while (True):
        new_matrix=merging(techniques,no_clust,s_matrix,k,alpha1)
        lamda = len(new_matrix)
        if (lamda>=needed_cluster):
            break
        else:
            alpha1 +=delta_alpha


    while len(new_matrix)>needed_cluster:
        new_matrix_2=(new_matrix!=0)*1
        new_s_matrix=similarity_matrix(len(new_matrix_2),[1]*len(new_matrix_2),new_matrix_2)
        alpha1=max(np.amax(new_s_matrix,axis=0))
        if alpha1<alpha_min:
            break
        else:
            new_merged_matrix=merging(len(new_matrix_2),[1]*len(new_matrix_2),new_s_matrix,new_matrix,alpha1)
        if len(new_merged_matrix)<=needed_cluster:
            break
        else:
            new_matrix=new_merged_matrix

  
    oss=object_similarity_score(new_merged_matrix)
    index=np.argmax(oss,axis=0)


    output1,cluster_with_object,uncertain_cluster=final_output(oss,alpha2,index)


    if (len(uncertain_cluster)==0):
        print(cluster_with_object)
    else:
        q_score = calculate_quality_score(output1)
        for iter in uncertain_cluster:
            indi_cluster = updated_quality_score(q_score,oss[index[iter]][iter],output1)
            output1[indi_cluster].append(oss[index[iter]][iter])
            cluster_with_object[indi_cluster].append(iter)
            q_score = calculate_quality_score(output1)
        #print(cluster_with_object)
        
        for i in range(len(cluster_with_object)):
            for j in range(len(cluster_with_object[i])):
                #print(j,i,cluster_with_object[i],cluster_with_object[i][j])
                clusters[cluster_with_object[i][j]]=i

    #end new main
    '''
    counter = Counter(clusters.tolist())

    for key in sorted(counter.keys()):
        print ("Label {0} has {1} samples".format(key, counter[key]))
        
    # create new hdf5 with clusters added
    with h5py.File(output_path, "w") as f:
        f.create_dataset("vectors", shape=vectors.shape, data=vectors)
        f.create_dataset("cluster", shape=(vectors.shape[0],), data=clusters, dtype=np.int32)
        f.create_dataset("notes", shape=(vectors.shape[0],), data=np.array(ips))

    print("ensemble algorithm applied")
