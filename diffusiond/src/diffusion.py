__author__ = 'decarlin'

import logging
from numpy import genfromtxt, dot, array
import math
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm, expm_multiply
import networkx as nx


class Diffuser:

    def __init__(self, networkN, time=0.1, normalize=True, input_vector=None,diffuse_key='DiffuseThis',calculate_kernel=False, normalize_laplacian=False):
        logging.info('Diffuser: Initializing')
        #TODO: Process networkN
        self.network = networkN
        self.time_T = time
        self.calculate_kernel=calculate_kernel
        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes_iter()]
        if normalize_laplacian:
            self.L=coo_matrix(nx.normalized_laplacian_matrix(nx.Graph(self.network)))
        else:
            self.L=coo_matrix(nx.laplacian_matrix(nx.Graph(self.network)))

        if isinstance(input_vector, list):
            self.input_vector=array([n in input_vector for n in self.node_names])
        elif isinstance(input_vector, dict):
            self.input_vector=array([input_vector[n] if n in input_vector.keys() else 0 for n in self.node_names])
        elif diffuse_key in self.network.node[0].keys():
            self.input_vector=array([n[diffuse_key] for n in self.network.node.values()])

        #self.input_vector=node_attr('DiffuseThisColumn',normalize=self.normalize)
        logging.info('Diffuser: Initialization complete')

    def start(self):
        """Diffuses the selected nodes against the network"""
        logging.info('Diffuser: Starting diffusion')
        #TODO: Perform diffusion
        if self.calculate_kernel:
            logging.info('Diffuser: Calculating kernel')
            self.calculateKernel(self.L)
            
        if self.input_vector is not None:
            if self.calculate_kernel:
                self.out_vector=self.kernel.dot(self.input_vector)
            else:
                self.out_vector=expm_multiply(-self.L,self.input_vector,start=0,stop=0.1,endpoint=True)[-1]
            
            node_dict=dict([(self.network.node.keys()[i],self.out_vector[i]) for i in range(len(self.network.node.keys()))])
            nx.set_node_attributes(self.network,'DiffusedOutput',node_dict)
        logging.info('Diffuser: Diffusion completed')
        return self.network

    def calculateKernel(self,L):
        self.kernel = expm(-self.time_T*L)

    def writeHeatSimilarity(self, filename, cutoff=0.0001):
        if self.kernel is not None:
            fh=open(filename,'w')

            for i in xrange(self.kernel.shape[0]):
                for j in range(i+1,dif.kernel.shape[1]):
                    if self.kernel[i,j]>cutoff:
                        fh.write('\t'.join([self.node_names[i],self.node_names[j],str(self.kernel[i,j]*1000)])+'\n')
            fh.close()
        else:
            warn('No kernel calcualted')
