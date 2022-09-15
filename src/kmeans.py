import pynq
import sys
import numpy as np
import random
import time
from datetime import timedelta
from math import ceil

class KmeansHLS(pynq.DefaultIP):
    """
        Driver for kmeans HLS kernel.
    """
    
    bindto  = ["xilinx.com:hls:workload:1.0"]
    
    def __init__(self,description):
        super().__init__(description=description)
        self._fullpath = description['fullpath']
        self.cluster_buffer = []
        self.input_buffer = []
        self.output_buffer = []
        self.aloc_time = 0.0
        self.copy_to_time = 0.0
        self.exec_time = 0.0
        self.copy_from_time = 0.0
        self.k = 0
        self.n = 0
    
    def allocate(self, k, n, num_points, dtype,ol):
        self.cluster_buffer = pynq.allocate((k*n,),dtype=dtype)    
        self.input_buffer = pynq.allocate((num_points*n,),dtype=dtype)
        self.output_buffer = pynq.allocate((num_points,),dtype=np.uint32)
        
#         self.cluster_buffer = pynq.allocate((k*n,),dtype=dtype,target = ol.HBM8)    
#         self.input_buffer = pynq.allocate((num_points*n,),dtype=dtype,target = ol.HBM0)
#         self.output_buffer = pynq.allocate((num_points,),dtype=np.uint32,target = ol.HBM16)
        
    def classify(self, k, n, clusters, data, ol): 
        self.k = k
        self.n = n
        
        st = time.time()
        self.allocate(k, n, len(data),data.dtype, ol)
        self.aloc_time = (time.time() - st) * 1000.0
             
        self.input_buffer[:] =data.flatten()
        self.cluster_buffer[:] = clusters.flatten()
        
        st = time.time()   
        self.input_buffer.sync_to_device() 
        self.cluster_buffer.sync_to_device()
        self.copy_to_time = (time.time() - st) * 1000.0
        
        st = time.time()
        self.call(self.cluster_buffer,self.input_buffer, self.output_buffer)
        self.exec_time = (time.time() - st) * 1000.0
        
        st = time.time()
        self.output_buffer.sync_from_device()
        self.copy_from_time = (time.time() - st) * 1000.0
               
        return np.array(self.output_buffer,dtype=np.uint32)
    
    def freebuffer(self):
        self.input_buffer.freebuffer()
        self.cluster_buffer.freebuffer()
        self.output_buffer.freebuffer()
        del self.input_buffer
        del self.cluster_buffer
        del self.output_buffer
    
    def print_stats(self):
        print('%s,%s,%s,%s,%s,%s'%('clusters','dimensions','alloc_time(ms)','copy_to_time(ms)','copy_from_time(ms)','exec_time(ms)'))
        print('%d, %d, %5.2f,%5.2f,%5.2f,%5.2f'%(self.k,self.n,self.aloc_time,self.copy_to_time,self.copy_from_time,self.exec_time))
        
    
class KmeansHw(pynq.DefaultIP):
    """
        Driver for kmeans RTL kernel.
    """
    
    bindto  = ["xilinx.com:RTLKernel:kernel_top:1.0"]
    
    def __init__(self,description):
        super().__init__(description=description)
        self._fullpath = description['fullpath']
        self.input_buffer = []
        self.output_buffer = []
        self.aloc_time = 0.0
        self.copy_to_time = 0.0
        self.exec_time = 0.0
        self.copy_from_time = 0.0
        self.k = 0
        self.n = 0
    
    def allocate(self, k, n, num_points, ol):
        num_points = num_points//4
        num_config_bytes = 64
        num_cluster_bytes = int(ceil((k*n*8.0)/64.0)*64.0)
        num_points_bytes = int(ceil((num_points * n * 2.0)/64.0)*64.0)
        total_in_bytes = num_config_bytes+num_cluster_bytes+num_points_bytes
        total_out_bytes = int(ceil(num_points_bytes/(2 * n)/64.0)*64.0)
        self.input_buffer0 = pynq.allocate((total_in_bytes,),dtype=np.byte, target=ol.HBM0)
        self.output_buffer0 = pynq.allocate((total_out_bytes,),dtype=np.byte,target=ol.HBM0)
        self.input_buffer1 = pynq.allocate((total_in_bytes,),dtype=np.byte,target=ol.HBM1)
        self.output_buffer1 = pynq.allocate((total_out_bytes,),dtype=np.byte,target=ol.HBM1)
        self.input_buffer2 = pynq.allocate((total_in_bytes,),dtype=np.byte, target=ol.HBM2)
        self.output_buffer2 = pynq.allocate((total_out_bytes,),dtype=np.byte,target=ol.HBM2)
        self.input_buffer3 = pynq.allocate((total_in_bytes,),dtype=np.byte,target=ol.HBM3)
        self.output_buffer3 = pynq.allocate((total_out_bytes,),dtype=np.byte,target=ol.HBM3)
        
    def flat_clusters(self,c):
        flat_c = np.array(c,dtype='u4').flatten()
        flat_idx = np.array([i for i in range(1,len(flat_c)+1)],dtype='u4')
        flat_with_idx_clusters = np.array(list(zip(flat_idx,flat_c)),dtype='u4').flatten().tobytes()
        return list(flat_with_idx_clusters)
    
    def flat_data(self, d):
        return list(np.array(d,dtype='u2').flatten().tobytes())
    
    def classify(self, k, n, clusters, data, ol): 
        self.k = k
        self.n = n
        flat_with_idx_clusters = self.flat_clusters(clusters)  
        flat_data = self.flat_data(data)
        
        quarter = len(flat_data)//4
       
        st = time.time()
        self.allocate(k, n, len(data),ol) 
        self.aloc_time = (time.time()-st)*1000.0
          
        
        self.input_buffer0[0:64] = list(int(k * n).to_bytes(64,sys.byteorder)) 
        self.input_buffer0[64:64+len(flat_with_idx_clusters)] = flat_with_idx_clusters    
        self.input_buffer0[64+len(flat_with_idx_clusters):64+len(flat_with_idx_clusters)+len(flat_data)] = flat_data[:quarter]
        
        self.input_buffer1[0:64] = list(int(k * n).to_bytes(64,sys.byteorder))
        self.input_buffer1[64:64+len(flat_with_idx_clusters)] = flat_with_idx_clusters    
        self.input_buffer1[64+len(flat_with_idx_clusters):64+len(flat_with_idx_clusters)+len(flat_data)] = flat_data[quarter:quarter*2]
        
        self.input_buffer1[0:64] = list(int(k * n).to_bytes(64,sys.byteorder))
        self.input_buffer1[64:64+len(flat_with_idx_clusters)] = flat_with_idx_clusters    
        self.input_buffer1[64+len(flat_with_idx_clusters):64+len(flat_with_idx_clusters)+len(flat_data)] = flat_data[quarter*2:quarter*3]
        
        
        self.input_buffer1[0:64] = list(int(k * n).to_bytes(64,sys.byteorder))
        self.input_buffer1[64:64+len(flat_with_idx_clusters)] = flat_with_idx_clusters    
        self.input_buffer1[64+len(flat_with_idx_clusters):64+len(flat_with_idx_clusters)+len(flat_data)] = flat_data[quarter*3:]
        
        
        st = time.time()
        self.input_buffer0.sync_to_device()
        self.input_buffer1.sync_to_device()    
        self.input_buffer2.sync_to_device()
        self.input_buffer3.sync_to_device()
        self.copy_to_time = (time.time()-st) * 1000.0
        
        st = time.time()
        self.call(len(self.input_buffer0),len(self.input_buffer1),len(self.input_buffer2),len(self.input_buffer3),len(self.output_buffer0),len(self.output_buffer1),len(self.output_buffer2),len(self.output_buffer3),self.input_buffer0, self.output_buffer0,self.input_buffer1, self.output_buffer1,self.input_buffer2, self.output_buffer2,self.input_buffer3, self.output_buffer3)
        self.exec_time = (time.time()-st) * 1000.0
        
        st = time.time()
        self.output_buffer0.sync_from_device()
        self.output_buffer1.sync_from_device()
        self.output_buffer2.sync_from_device()
        self.output_buffer3.sync_from_device()
        self.copy_from_time = (time.time()-st) * 1000.0    
        
        return np.concatenate((self.output_buffer0,self.output_buffer1,self.output_buffer2,self.output_buffer3))
    
    def print_stats(self):
        print('%s,%s,%s,%s,%s,%s'%('clusters','dimensions','alloc_time(ms)','copy_to_time(ms)','copy_from_time(ms)','exec_time(ms)'))
        print('%d, %d, %5.2f,%5.2f,%5.2f,%5.2f'%(self.k,self.n,self.aloc_time,self.copy_to_time,self.copy_from_time,self.exec_time))
       
    
class KMeansFPGA():
    def __init__(self, n_clusters, n_dims, max_iter = 10, xclbin=''):
        self._xclbin = xclbin
        self._n_clusters = n_clusters
        self._n_dims = n_dims
        self._max_iter = max_iter
        self.ol = pynq.Overlay(xclbin)
        self.kmeans_hw = self.ol.kernel_top_1
        
    def fit(self, X):
        self.clusters = [ [j if i == 0 else 0 for i in range(self._n_dims)] for j in range(self._n_clusters) ]     
        clusters_old = self.clusters
        
        pred = self.kmeans_hw.classify(self._n_clusters,self._n_dims,self.clusters,X)
        
        flat_data = np.array(X).flatten()
        num_point = len(X)
        
        k_sum = np.ndarray((self._n_clusters*self._n_dims,), dtype=int)
        k_avg = np.ndarray((self._n_clusters,), dtype=int)
              
        for it in range(self._max_iter):
            k_sum.fill(0)
            k_avg.fill(0)
            for i in range(num_point):
                for j in range(self._n_dims):
                     k_sum[pred[i] * self._n_dims + j] += flat_data[i * self._n_dims + j]
                k_avg[pred[i]] += 1
            
            different = 0
            for j in range(self._n_clusters):
                for d in range(self._n_dims):
                    idx = j * self._n_dims + d
                    if k_avg[idx // self._n_dims] > 0:
                        self.clusters[j][d] = int(k_sum[idx] // k_avg[idx // self._n_dims])
      
                    if self.clusters[j][d] != clusters_old[j][d]:
                        different = 1;

                    clusters_old[j][d] = self.clusters[j][d];
            
            if different == 0:
                pred = self.kmeans_hw.classify(self._n_clusters,self._n_dims,self.clusters,[])
            else:
                break
                
        return self
    
     
    def predict(self, C=[], X=[]):
        if len(C) > 0:
            self.clusters = C
            
        pred = self.kmeans_hw.classify(self._n_clusters,self._n_dims,self.clusters,X)
        
        return np.array(pred[:len(X)],dtype=int)
        
    def free(self):
        pynq.Overlay.free(self.ol)
    
    def print_stats(self):
        self.kmeans_hw.print_stats()