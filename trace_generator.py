import argparse
import os
import sys
import json
import sympy
from dataclasses import dataclass, field
from typing import List, Dict, Any, ClassVar, Set
from itertools import product
import numpy as np

L = sympy.symbols("L")
H_k = sympy.symbols("H_k")
H_q = sympy.symbols("H_q")
H_g = sympy.symbols("H_g")
D = sympy.symbols("D")
D_h = sympy.symbols("D_h")
D_ff = sympy.symbols("D_ff")
dp = sympy.symbols("dp")
pp = sympy.symbols("pp")
cp = sympy.symbols("cp")
tp = sympy.symbols("tp")
B = sympy.symbols("B")
S = sympy.symbols("S")
M = sympy.symbols("M")

SUBSTITUTE_VALUES = {
    L: 24,
    H_k: 12,
    H_q: 12,
    H_g: 12,
    D_h: 128,
    D: 512,
    D_ff: 512,
    dp: 2,
    pp: 2,
    cp: 2,
    tp: 4,
    B: 4,
    S: 4,
    M: 4
}

MAX_COMM_GROUPS_PER_COMM = 1

@dataclass
class Node:

    _next_id: ClassVar[int] = 0
    _all_nodes: ClassVar[Dict[int, "Node"]] = {}

    id:int = field(init=False)
    type:str
    subtype:str
    name:str
    comm_group:List
    size:int
    deps:List

    def __post_init__(self):
        self.id = Node._next_id
        Node._all_nodes[self.id] = self
        Node._next_id += 1

class CommGroupTracker:
    def __init__(self):
        self.comm_groups:Dict[Set[int], int] = {}

    def exists(self, nodes:List[int]|Set[int]) -> bool:
        if isinstance(nodes, list):
            nodes = set(nodes)
        return nodes in self.comm_groups

    def fetch(self, nodes:List[int]|Set[int]) -> int:
        if isinstance(nodes, list):
            nodes = set(nodes)
        return self.comm_groups[nodes]

    def add(self, nodes:List[int]|Set[int]) -> bool:
        if isinstance(nodes, list):
            nodes = set(nodes)
        if nodes in self.comm_groups:
            return False
        else:
            new_id = len(self.comm_groups)
            self.comm_groups[nodes] = new_id
            return True

sharding_formulas = {
    "fwd_cp_all_gather_k" : M*S/cp*H_k/tp*D_h,
    "fwd_cp_all_gather_v" : M*S/cp*H_k/tp*D_h,
    "fwd_tp_all_reduce_attn" : M*S/cp*D,
    "fwd_tp_all_reduce_ffn" : M*S/cp*D,
    "fwd_fsdp_all_gather" : M*S/cp*D,
}

def pipeline_stage_from_layer_id(layer_id:int,num_layers:int) -> int:
    num_pipeline_stages = pp.subs(SUBSTITUTE_VALUES)
    layers_per_stage = num_layers // num_pipeline_stages
    return layer_id // layers_per_stage

def node_id_from_axes(d,c=0,p=0,t=0):
    DP=dp.subs(SUBSTITUTE_VALUES)
    PP=pp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)
    if isinstance(d, int):
        #All inputs are integers
        pass
    elif isinstance(d, list) or isinstance(d, np.ndarray):
        #All inputs are lists or arrays
        c = d[1]
        p = d[2]
        t = d[3]
        d = d[0]
    return d*PP*CP*TP + p*CP*TP + c*TP + t

def axes_from_node_id(node_id:int):
    DP=dp.subs(SUBSTITUTE_VALUES)
    PP=pp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)

    d = node_id // (PP*CP*TP)
    p = (node_id % (PP*CP*TP)) // (CP*TP)
    c = (node_id % (CP*TP)) // TP
    t = node_id % TP

    return d,p,c,t

def permute_axes(d:int,c:int,p:int,t:int)->List[List[int]]:
    DP=dp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    PP=pp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)


    @dataclass
    class Axis:
        name:str = ""
        type:str = ""
        degree:int = 0
        values:List[int] = field(default_factory=list)

    def expand(name:str,p:int,P:int)->Axis:
        if p == -1:
            return Axis(name,"sweep",P,list(range(P)))
        elif p == -2:
            return Axis(name,"collect",P,list(range(P)))
        else:
            return Axis(name,"fixed",P,[p])

    dp_axis = expand("dp",d,DP)
    cp_axis = expand("cp",c,CP)
    pp_axis = expand("pp",p,PP)
    tp_axis = expand("tp",t,TP)

    axes = {ax.name:ax for ax in [dp_axis,cp_axis,pp_axis,tp_axis]}

    collect_axis = [ax for ax in axes.values() if ax.type == "collect"]
    assert len(collect_axis) == 1, f"Only one axis can be collect, found {len(collect_axis)}"

    permute_axes = [ax for ax in axes.values() if ax.type != "collect"]

    scrambled_dims = [ax.name for ax in permute_axes] + [collect_axis[0].name]

    scrambled_ordered_coords = list(product(*([ax.values for ax in permute_axes] + [collect_axis[0].values])))

    matrix_scrambled_coords = np.array(scrambled_ordered_coords)

    #Split into columns
    cols:Dict[str,np.ndarray] = {name:matrix_scrambled_coords[:,idx] for idx,name in enumerate(scrambled_dims)}

    # Form into new item based on desired ordering
    ordered_cols = [cols[name] for name in ["dp","cp","pp","tp"]]
    matrix_ordered_coords = np.column_stack(ordered_cols)

    correctly_shaped_matrix_coords = np.reshape(matrix_ordered_coords,[-1,collect_axis[0].degree,4])
    # print(correctly_shaped_matrix_coords)

    # for i in range(correctly_shaped_matrix_coords.shape[0]):
    #     for j in range(correctly_shaped_matrix_coords.shape[1]):
    #         d,c,p,t = correctly_shaped_matrix_coords[i,j]
    #         node_id = node_id_from_axes(d,c,p,t)
    #         print(f"{d},{c},{p},{t} -> {node_id}")
    
    npu_id_matrix = np.apply_along_axis(node_id_from_axes, -1, correctly_shaped_matrix_coords)    
    # print(npu_id_matrix.tolist())

    return npu_id_matrix.tolist()
    

def single_layer_forward_pass(layer_id,microbatch_id):
    #FSDP fetch is outside the scope of this function

    pipeline_stage_id = pipeline_stage_from_layer_id(layer_id,L.subs(SUBSTITUTE_VALUES))

    pre_layer_sync_node = Node(
        "SYNC",
        "PRE_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.pre_layer_sync",
        [],
        0,
        []
    )

    #If this is the first microbatch, then we need to fetch weights across fsdp
    if microbatch_id == 0:
        fsdp_all_gather_comm_groups = permute_axes(-2,-1,pipeline_stage_id,-1)
        fsdp_all_gather_nodes = []
        for comm_group in fsdp_all_gather_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(fsdp_all_gather_comm_groups))]:
            all_gather = Node(
                "COMM",
                "ALL_GATHER",
                f"mb{microbatch_id}.layer{layer_id}.fwd.fsdp_all_gather",
                comm_group,
                sharding_formulas["fwd_fsdp_all_gather"].subs(SUBSTITUTE_VALUES),
                [pre_layer_sync_node.id]
            )
            fsdp_all_gather_nodes.append(all_gather)

    post_fsdp_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.post_fsdp_all_gather",
        [],
        0,
        [node.id for node in fsdp_all_gather_nodes] if microbatch_id == 0 else [pre_layer_sync_node.id]
    )

    #If we are doing all gather across CP and for a fixed PP, then there will be an equivalent all gather across each TP and DP.
    #So there are a total of DP*TP all gather operations across CP for each layer and microbatch. 
    #We can get the comm groups for each of these all gather operations by permuting the axes
    cp_all_gather_comm_groups = permute_axes(-1,-2,pipeline_stage_id,-1)
    cp_all_gather_nodes = []
    for comm_group in cp_all_gather_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(cp_all_gather_comm_groups))]:
        #All gather K across CP
        all_gather_k = Node(
            "COMM",
            "ALL_GATHER",
            f"mb{microbatch_id}.layer{layer_id}.fwd.cp_all_gather.k",
            comm_group,
            sharding_formulas["fwd_cp_all_gather_k"].subs(SUBSTITUTE_VALUES),
            [post_fsdp_sync_node.id]
        )
        all_gather_v = Node(
            "COMM",
            "ALL_GATHER",
            f"mb{microbatch_id}.layer{layer_id}.fwd.cp_all_gather.v",
            comm_group,
            sharding_formulas["fwd_cp_all_gather_v"].subs(SUBSTITUTE_VALUES),
            [post_fsdp_sync_node.id]
        )
        cp_all_gather_nodes.append(all_gather_k)
        cp_all_gather_nodes.append(all_gather_v)

    cp_all_gather_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.post_cp_all_gather",
        [],
        0,
        [node.id for node in cp_all_gather_nodes]
    )

    tp_all_reduce_comm_groups = permute_axes(-1,-1, pipeline_stage_id,-2)
    tp_all_reduce_attn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:
        all_reduce_attn = Node(
            "COMM",
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.fwd.tp_all_reduce.attn",
            comm_group,
            sharding_formulas["fwd_tp_all_reduce_attn"].subs(SUBSTITUTE_VALUES),
            [cp_all_gather_sync_node.id]
        )
        tp_all_reduce_attn_nodes.append(all_reduce_attn)

    tp_all_reduce_attn_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.post_tp_all_reduce.attn",
        [],
        0,
        [node.id for node in tp_all_reduce_attn_nodes]
    )
    
    tp_all_reduce_ffn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:
        all_reduce_ffn = Node(
            "COMM",
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.fwd.tp_all_reduce.ffn",
            comm_group,
            sharding_formulas["fwd_tp_all_reduce_ffn"].subs(SUBSTITUTE_VALUES),
            [tp_all_reduce_attn_sync_node.id]
        )
        tp_all_reduce_ffn_nodes.append(all_reduce_ffn)

    post_layer_sync_node = Node(
        "SYNC",
        "POST_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.post_layer_sync",
        [],
        0,
        [node.id for node in tp_all_reduce_ffn_nodes]
    )

    return [pre_layer_sync_node,post_layer_sync_node]

def single_layer_backward_pass(layer_id,microbatch_id):
    #FSDP reduce scatter is outside the scope of this function
    
    pre_layer_sync_node = Node(
        "SYNC",
        "PRE_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.pre_layer_sync",
        [],
        0,
        []
    )

    tp_all_reduce_comm_groups = permute_axes(-1,-1, pipeline_stage_from_layer_id(layer_id,L.subs(SUBSTITUTE_VALUES)),-2)
    tp_all_reduce_ffn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:
        all_reduce_ffn = Node(
            "COMM",
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.bwd.tp_all_reduce.ffn",
            comm_group,
            sharding_formulas["fwd_tp_all_reduce_ffn"].subs(SUBSTITUTE_VALUES),
            [pre_layer_sync_node.id]
        )
        tp_all_reduce_ffn_nodes.append(all_reduce_ffn)

    tp_all_reduce_ffn_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.post_tp_all_reduce.ffn",
        [],
        0,
        [node.id for node in tp_all_reduce_ffn_nodes]
    )

    cp_reduce_scatter_comm_groups = permute_axes(-1,-2,pipeline_stage_from_layer_id(layer_id,L.subs(SUBSTITUTE_VALUES)),-1)
    cp_reduce_scatter_nodes = []
    for comm_group in cp_reduce_scatter_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(cp_reduce_scatter_comm_groups))]:
        reduce_scatter_k = Node(
            "COMM",
            "REDUCE_SCATTER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.cp_reduce_scatter.k",
            comm_group,
            sharding_formulas["fwd_cp_all_gather_k"].subs(SUBSTITUTE_VALUES),
            [tp_all_reduce_ffn_sync_node.id]
        )
        reduce_scatter_v = Node(
            "COMM",
            "REDUCE_SCATTER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.cp_reduce_scatter.v",
            comm_group,
            sharding_formulas["fwd_cp_all_gather_k"].subs(SUBSTITUTE_VALUES),
            [tp_all_reduce_ffn_sync_node.id]
        )
        cp_reduce_scatter_nodes.append(reduce_scatter_k)
        cp_reduce_scatter_nodes.append(reduce_scatter_v)

    cp_reduce_scatter_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.post_cp_reduce_scatter",
        [],
        0,
        [node.id for node in cp_reduce_scatter_nodes]
    )

    #Reuse same comm  groups as before
    tp_all_reduce_attn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:

        all_reduce_attn = Node(
            "COMM",
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.bwd.tp_all_reduce.attn",
            comm_group,
            sharding_formulas["fwd_tp_all_reduce_attn"].subs(SUBSTITUTE_VALUES),
            [cp_reduce_scatter_sync_node.id]
        )
        tp_all_reduce_attn_nodes.append(all_reduce_attn)

    post_layer_sync_node = Node(
        "SYNC",
        "POST_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.post_layer_sync",
        [],
        0,
        [node.id for node in tp_all_reduce_attn_nodes]
    )

    return [pre_layer_sync_node,post_layer_sync_node]

def forward_pipeline_stage(layer_ids,pipeline_stage_id,microbatch_id):
    print(f"Received layer_ids for forward pass({pipeline_stage_id}): {layer_ids}")
    layers = []
    L = len(layer_ids)
    for layer_id in layer_ids:
        layers.append(single_layer_forward_pass(layer_id,microbatch_id))

    #Add a pre pipeline sync
    pre_pipeline_sync_node = Node(
        "SYNC",
        "PRE_PIPELINE",
        f"mb{microbatch_id}.pipeline_stage{pipeline_stage_id}.fwd.pre_pipeline_sync",
        [],
        0,
        []
    )

    #Connect first layers pre_layer_sync to pre_pipeline_sync
    layers[0][0].deps.append(pre_pipeline_sync_node.id)

    #Wire the layers together by connecting this layers pre_layer_sync to previous layers post_layer_sync
    #For example, connect layer 14's pre_layer_sync to layer 15's post_layer_sync
    for layer_id in layer_ids[1:]:
        print(f"L;{layer_id}/{layer_id%L} -> L-1;{layer_id-1}/{layer_id%L-1}")
        layers[layer_id%L][0].deps.append(layers[layer_id%L-1][1].id)

    post_pipeline_sync_node = Node(
        "SYNC",
        "POST_PIPELINE",
        f"mb{microbatch_id}.pipeline_stage{pipeline_stage_id}.fwd.post_pipeline_sync",
        [],
        0,
        [layers[-1][1].id]
    )

    return pre_pipeline_sync_node, post_pipeline_sync_node

def backward_pipeline_stage(layer_ids,pipeline_stage_id,microbatch_id):
    #I am expecting layer ids to already be in reverse order like [15,14,13,12]
    print(f"Received layer_ids for backward pass({pipeline_stage_id}): {layer_ids}")
    layers = []
    L=len(layer_ids)
    for layer_id in layer_ids:
        layers.append(single_layer_backward_pass(layer_id,microbatch_id))

    #Add a pre pipeline sync
    pre_pipeline_sync_node = Node(
        "SYNC",
        "PRE_PIPELINE",
        f"mb{microbatch_id}.pipeline_stage{pipeline_stage_id}.bwd.pre_pipeline_sync",
        [],
        0,
        []
    )

    #Connect first layers pre_layer_sync to pre_pipeline_sync
    layers[0][0].deps.append(pre_pipeline_sync_node.id)

    #Wire the layers by connecting current layers this layers all_reduce_ffn to previous layers all_reduce_attn
    #For example, connect layer 14's all_reduce_ffn to layer 15's all_reduce_attn
    for layer_id in layer_ids[1:]:
        print(f"L;{layer_id+1}/{layer_id%L+1} -> L;{layer_id}/{layer_id%L}")
        print(f"Connecting layer {layer_id%L+1}'s post_layer_sync to layer {layer_id%L}'s pre_layer_sync")
        layers[layer_id%L+1][0].deps.append(layers[layer_id%L][1].id)

    #Add a post pipeline sync
    post_pipeline_sync_node = Node(
        "SYNC",
        "POST_PIPELINE",
        f"mb{microbatch_id}.pipeline_stage{pipeline_stage_id}.bwd.post_pipeline_sync",
        [],
        0,
        [layers[-1][1].id]
    )

    return pre_pipeline_sync_node, post_pipeline_sync_node

def pipeline_pass_for_single_microbatch(microbatch_id,num_layers):
    num_pipeline_stages = pp.subs(SUBSTITUTE_VALUES)
    layers_per_stage = num_layers // num_pipeline_stages

    fwd_stages = []
    for i in range(num_pipeline_stages):
        layer_ids = list(range(i*layers_per_stage, (i+1)*layers_per_stage))
        fwd_stages.append(forward_pipeline_stage(layer_ids,i,microbatch_id))

    #Wire them up
    for i in range(1,num_pipeline_stages):
        fwd_stages[i][0].deps.append(fwd_stages[i-1][1].id)

    bwd_stages = []
    for i in range(num_pipeline_stages):
        layer_ids = list(reversed(range((num_pipeline_stages-i-1)*layers_per_stage, (num_pipeline_stages-i)*layers_per_stage)))
        bwd_stages.append(backward_pipeline_stage(layer_ids,num_pipeline_stages-i-1,microbatch_id))

    #Wire them up
    for i in range(1,num_pipeline_stages):
        bwd_stages[i][0].deps.append(bwd_stages[i-1][1].id)

    # Connect forward and backward passes
    bwd_stages[0][0].deps.append(fwd_stages[-1][1].id)

    return fwd_stages, bwd_stages

def write_trace_to_json(nodes:List[Node], filepath:str) -> None:
    #Ignoring comm group deduplication (CommGroupTracker) for now — comm_group is written
    #as the literal list of involved NPU ids per node, for visual inspection.
    trace = [
        {
            "id": node.id,
            "type": node.type,
            "subtype": node.subtype,
            "name": node.name,
            "comm_group": [int(npu_id) for npu_id in node.comm_group],
            "size": int(node.size),
            "deps": node.deps,
        }
        for node in sorted(nodes, key=lambda n: n.id)
    ]
    json_obj = {
        "nodes": trace,
    }
    with open(filepath, "w") as f:
        json.dump(json_obj, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a analytical model compatible DAG from model and training config")

    #I'm fixing the ordering for sharding axes based on how deep inside the model they are located
    #dp replicates entire model
    #cp is at sequence level
    #pp happens at multi layer level
    #tp and ep happen at sublayer level
    sharding_axes_symbols = [dp,cp,pp,tp]
    sharding_axes = [s.subs(SUBSTITUTE_VALUES) for s in sharding_axes_symbols]

    # fwd_layers = forward_pipeline_stage([0,1,2,3],0,0)
    # bwd_layers = backward_pipeline_stage([3,2,1,0],0,0)

    # # print(bwd_layers[0][0].name)
    # # print(fwd_layers[1][1].name)
    # bwd_layers[0][0].deps.append(fwd_layers[-1][1].id)

    pipeline_pass_for_single_microbatch(0,8)

    all_nodes_single_layer_single_microbatch = Node._all_nodes

    write_trace_to_json(list(all_nodes_single_layer_single_microbatch.values()), "trace_single_layer.json")
