import torch
from torch_geometric.data import Data
import numpy as np
import pickle
import argparse
import os
import pathlib

def generate_noCycles(Nsamples,d, **kwargs):
    """
    Dataset where label = 1: cycle
    label = 0 : no cycle
    """

    labels = np.random.randint(2,size = Nsamples)
    x_list = []
    edge_list = []
    for n in range(Nsamples):
        Nnodes = np.random.randint(10,20)
        if labels[n]:
            edge_index = torch.stack((torch.arange(Nnodes-1),(1+torch.arange(Nnodes-1))))
            x = torch.randn(Nnodes,d)
        else:
            edge_index = torch.stack((torch.arange(Nnodes),(1+torch.arange(Nnodes))%Nnodes))
            x = torch.randn(Nnodes,d)

        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),1)
        

        x_list += [x]
        edge_list += [edge_index]
            
            
    with open("./NoCycles/graphs.txt", "wb") as fp:
        pickle.dump([x_list, edge_list], fp)
    torch.save(torch.tensor(labels),"./NoCycles/labels.pt")



def generate_dummy(Nsamples,d, **kwargs):

    x_list = []
    edge_list = []

    labels = np.random.randint(2,size=Nsamples)
    for n in range(Nsamples):

        Nnodes = 4
        if labels[n]:
            edge_index = torch.tensor([[0,0,0,1,1,2],[1,2,3,2,3,3]])
        else:
            edge_index = torch.tensor([[0,0,1,2],[1,2,3,3]])

        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),1)
        x = torch.randn(Nnodes,d)

        x_list += [x]
        edge_list += [edge_index]


    with open("./Dummy/graphs.txt", "wb") as fp:
        pickle.dump([x_list, edge_list], fp)
    torch.save(torch.tensor(labels),"./Dummy/labels.pt")



def generate_necklaces(Nsamples,d, **kwargs):

    small_chain_length = 2

    labels = np.random.randint(2,size=Nsamples)
    x_list = []
    edge_list = []
    for n in range(Nsamples):
        Nnodes = np.random.randint(10,20)

        nodes_chain = Nnodes - 2*small_chain_length
        chain_edge_index = torch.stack((torch.arange(nodes_chain-1),1+torch.arange(nodes_chain-1)))

        connect_point1 = int(nodes_chain/3)
        connect_point2 = int(2*nodes_chain/3)

        small_chain1_min_node = nodes_chain
        small_chain2_min_node = small_chain1_min_node + small_chain_length

        small_chain1 = small_chain1_min_node + torch.stack((torch.arange(small_chain_length-1),1+torch.arange(small_chain_length-1)))
        small_chain2 = small_chain2_min_node + torch.stack((torch.arange(small_chain_length-1),1+torch.arange(small_chain_length-1)))

        
        if labels[n]:

            connection_with_chain = torch.tensor([[connect_point1,connect_point1, connect_point2, connect_point2],[small_chain1_min_node, small_chain1_min_node + small_chain_length - 1, small_chain2_min_node, small_chain2_min_node+small_chain_length-1]])
 
        else:
            connection_with_chain = torch.tensor([[connect_point1,connect_point1, connect_point2, connect_point2],[small_chain1_min_node, small_chain2_min_node, small_chain1_min_node + small_chain_length - 1, small_chain2_min_node+small_chain_length-1]])
        
        edge_index = torch.cat((chain_edge_index,small_chain1,small_chain2,connection_with_chain),1)

        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),1)
        x = torch.randn(Nnodes,d)

        x_list += [x]
        edge_list += [edge_index]


    with open("./Necklaces/graphs.txt", "wb") as fp:
        pickle.dump([x_list, edge_list], fp)
    torch.save(torch.tensor(labels),"./Necklaces/labels.pt")



def generate_cycles(Nsamples,d, min_cycle = 3,**kwargs):

    labels = np.random.randint(2,size = Nsamples)
    x_list = []
    edge_list = []
    for n in range(Nsamples):
        Nnodes = np.random.randint(10,20)
        if labels[n]:
            n_triangles = (Nnodes-min_cycle) // min_cycle
            extra_cycle_length =  Nnodes - n_triangles * min_cycle

            edge_index_list = []
            for tri in range(n_triangles):
                edge_index_list += [ tri * min_cycle + torch.stack((torch.arange(min_cycle),(1+torch.arange(min_cycle))%min_cycle))]
            edge_index_list += [n_triangles * min_cycle + torch.stack((torch.arange(extra_cycle_length),(1+torch.arange(extra_cycle_length))%extra_cycle_length))]

            edge_index = torch.cat(edge_index_list,axis=1)
            x = torch.randn(Nnodes,d)
        else:
            edge_index = torch.stack((torch.arange(Nnodes),(1+torch.arange(Nnodes))%Nnodes))
            x = torch.randn(Nnodes,d)
        
        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),1)
        x_list += [x]
        edge_list += [edge_index]
            

    os.makedirs(f"./Cycles_{min_cycle}/", exist_ok=True)
            
    with open(f"./Cycles_{min_cycle}/graphs.txt", "wb") as fp:
        pickle.dump([x_list, edge_list], fp)
    torch.save(torch.tensor(labels),f"./Cycles_{min_cycle}/labels.pt")

if __name__=="__main__":
    
    DATASET_CHOICES = {"Dummy":generate_dummy,
            "Necklaces":generate_necklaces,
            "Cycle":generate_cycles,
            "NoCycle":generate_noCycles
            }

    parser = argparse.ArgumentParser(description='Generation of Synthetic Graph Datasets')
    parser.add_argument('--dataset', type=str, choices = DATASET_CHOICES.keys(), default = "Dummy")
    parser.add_argument('--Nsamples',type=int,default = 1000)
    parser.add_argument('--d',type=int,default = 3, help="Number of dimensions of the node features")
    parser.add_argument('--min_cycle',type=int, default = 3, help = "Size of smallest cycle in the Cycles graph")

    args = parser.parse_args()

    DATASET_CHOICES[args.dataset](**vars(args))

    print(f"Congrats ! You just generated {args.Nsamples} {args.dataset} Graphs.")
