__author__ = 'decarlin'

import logging
from numpy import genfromtxt, dot, array
import ast
import math
import base64
import operator
from scipy.sparse import coo_matrix,csc_matrix
from scipy.sparse.linalg import expm, expm_multiply
import networkx as nx
import json

class Diffuser:

    def __init__(self, networkN, options):
        logging.info('Diffuser: Initializing')
        self.network = networkN
        self.time_T = options.get('time', 0.1)
        self.calculate_kernel = options.get('kernel', 'False') == 'True'
        input_vector= options.get('heatvector', '')
        diffuse_key= options.get('heatattribute', 'diffusion_input')
        normalize_laplacian = options.get('normalize', 'False') == 'True'

        if input_vector != '':
            input_vector = json.loads(base64.b64decode(input_vector))

        self.node_names = [self.network.node[n]['name'] for n in self.network.nodes_iter()]
        if normalize_laplacian:
            self.L=csc_matrix(nx.normalized_laplacian_matrix(nx.Graph(self.network)))
        else:
            self.L=csc_matrix(nx.laplacian_matrix(nx.Graph(self.network)))

        if isinstance(input_vector, list):
            self.input_vector=array([n in input_vector for n in self.node_names])
        elif isinstance(input_vector, dict):
            self.input_vector=array([input_vector[n] if n in input_vector.keys() else 0 for n in self.node_names])
        else:
            found_heat=False
            heat_list=list()
            self.input_vector=array([])
            for n in self.network.nodes():
                if diffuse_key in self.network.node[n].keys():
                    heat_list.append(self.network.node[n][diffuse_key])
                    found_heat=True
                else:
                    heat_list.append(0)
            if not found_heat:
                raise Exception('No input heat found')
            self.input_vector=array(heat_list)

        #self.input_vector=node_attr('DiffuseThisColumn',normalize=self.normalize)
        logging.info('Diffuser: Initialization complete')

    def start(self):
        """Diffuses the selected nodes against the network"""
        logging.info('Diffuser: Starting diffusion')

        if self.calculate_kernel:
            logging.info('Diffuser: Calculating kernel')
            self.calculateKernel(self.L)

        logging.info('Diffuser: Calculating kernel')

        if self.input_vector is not None:
            if self.calculate_kernel:
                self.out_vector=self.kernel.dot(self.input_vector)
            else:
                self.out_vector=expm_multiply(-self.L,self.input_vector,start=0,stop=0.1,endpoint=True)[-1]

            self.node_dict=dict([(self.network.node.keys()[i],self.out_vector[i]) for i in range(len(self.network.node.keys()))])
            sorted_diffused = sorted(self.node_dict.items(), key=operator.itemgetter(1), reverse=True)
            self.node_dict_rank=dict([(sorted_diffused[i][0],i) for i in range(len(sorted_diffused))])
            nx.set_node_attributes(self.network,'diffused_output',self.node_dict)
            nx.set_node_attributes(self.network,'diffused_output_rank',self.node_dict_rank)
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
            raise Exception('No kernel calcualted')
