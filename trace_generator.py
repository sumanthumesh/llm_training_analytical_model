import argparse
import os
import sys
import json
import sympy
from dataclasses import dataclass, field
from typing import List, Dict, Any, ClassVar, Set, Tuple
from itertools import product
import numpy as np
import enum

L = sympy.symbols("L")
H_k = sympy.symbols("H_k")
H_q = sympy.symbols("H_q")
H_g = H_q/H_k
D_h = sympy.symbols("D_h")
D = D_h*H_q
D_ff = sympy.symbols("D_ff")
V = sympy.symbols("V")
dp = sympy.symbols("dp")
pp = sympy.symbols("pp")
cp = sympy.symbols("cp")
tp = sympy.symbols("tp")
B = sympy.symbols("B")
S = sympy.symbols("S")
M = sympy.symbols("M")

SUBSTITUTE_VALUES = {
    L: 128,
    H_k: 8,
    H_q: 128,
    D_h: 128,
    D_ff: 53248,
    V: 128000,
    dp: 4,
    pp: 2,
    cp: 1,
    tp: 4,
    B: 32*4,
    S: 8192,
    M: 16
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

def make_comm_node(subtype:str, name:str, comm_group:List, size, deps:List, degree:int) -> Node:
    """Creates a COMM node for a real collective, or a DUMMY node if the
    parallelism dimension it's over has degree 1 (e.g. a CP all-gather when
    cp=1 has no other rank to gather from -- it's a no-op, not a collective).
    Keeps the original subtype for traceability; comm_group/size are emptied
    since there's nothing to communicate.
    """
    if degree == 1:
        return Node("DUMMY", subtype, name, [], 0, deps)
    return Node("COMM", subtype, name, comm_group, size, deps)

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

sharded_weight_sizes = {
    "W_k": D*H_k/tp*D_h,
    "W_v": D*H_k/tp*D_h,
    "W_q": D*H_q/tp*D_h,
    "W_oA": H_q/tp*D_h*D,
    "W_1": D*D_ff/tp,
    "W_2": D*D_ff/tp,
    "W_oF": D_ff/tp*D
}

def per_layer_weight_sizes():
    #Note these are already sharded across tp
    total = sum(sharded_weight_sizes.values())
    return total

def per_layer_gradient_sizes():
    #Assuming grdient size is same as weight size
    return per_layer_weight_sizes()

sharding_formulas = {
    "fwd_cp_all_gather_k" : M*S*H_k/tp*D_h,
    "fwd_cp_all_gather_v" : M*S*H_k/tp*D_h,
    "fwd_tp_all_reduce_attn" : M*S/cp*D,
    "fwd_tp_all_reduce_ffn" : M*S/cp*D,
    "fwd_fsdp_all_gather" : per_layer_weight_sizes(),
    "bwd_tp_all_reduce_ffn" : M*S/cp*D,
    "bwd_cp_reduce_scatter_k" : M*S*H_k/tp*D_h,
    "bwd_cp_reduce_scatter_v" : M*S*H_k/tp*D_h,
    "bwd_tp_all_reduce_attn" : M*S/cp*D,
    "bwd_fsdp_reduce_scatter" : per_layer_gradient_sizes(),
    "bwd_cp_all_reduce" : per_layer_gradient_sizes()/dp,
    "pipeline_transfer" : M*S/cp*D
}



def pipeline_stage_from_layer_id(layer_id:int,num_layers:int) -> int:
    num_pipeline_stages = pp.subs(SUBSTITUTE_VALUES)
    layers_per_stage = num_layers // num_pipeline_stages
    return layer_id // layers_per_stage

# def node_id_from_axes(d,c=0,p=0,t=0):
def node_id_from_axes(dims):
    DP=dp.subs(SUBSTITUTE_VALUES)
    PP=pp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)
    #All inputs are lists or arrays
    c = dims[1]
    p = dims[2]
    t = dims[3]
    d = dims[0]
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

def get_pp_transfer_comm_groups(from_stage:int,to_stage:int):
    TP=tp.subs(SUBSTITUTE_VALUES)
    DP=dp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)

    comm_group_coords = []

    for d in range(DP):
        for c in range(CP):
            for t in range(TP):
                comm_group_coords.append([(d,c,from_stage,t),(d,c,to_stage,t)])
    
    comm_group_node_ids = [[node_id_from_axes(coord) for coord in group] for group in comm_group_coords]
    return comm_group_node_ids


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
    DP=dp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)

    pre_layer_sync_node = Node(
        "SYNC",
        "PRE_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.pre_layer_sync",
        [],
        0,
        []
    )

    #If this is the first microbatch, then we need to fetch weights across fsdp
    fsdp_all_gather_nodes = []
    if microbatch_id == 0:
        fsdp_all_gather_comm_groups = permute_axes(-2,-1,pipeline_stage_id,-1)
        for comm_group in fsdp_all_gather_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(fsdp_all_gather_comm_groups))]:
            all_gather = make_comm_node(
                "ALL_GATHER",
                f"mb{microbatch_id}.layer{layer_id}.fwd.fsdp_all_gather",
                comm_group,
                sharding_formulas["fwd_fsdp_all_gather"].subs(SUBSTITUTE_VALUES),
                [pre_layer_sync_node.id],
                degree=DP
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
        all_gather_k = make_comm_node(
            "ALL_GATHER",
            f"mb{microbatch_id}.layer{layer_id}.fwd.cp_all_gather.k",
            comm_group,
            sharding_formulas["fwd_cp_all_gather_k"].subs(SUBSTITUTE_VALUES),
            [post_fsdp_sync_node.id],
            degree=CP
        )
        all_gather_v = make_comm_node(
            "ALL_GATHER",
            f"mb{microbatch_id}.layer{layer_id}.fwd.cp_all_gather.v",
            comm_group,
            sharding_formulas["fwd_cp_all_gather_v"].subs(SUBSTITUTE_VALUES),
            [post_fsdp_sync_node.id],
            degree=CP
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
        all_reduce_attn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.fwd.tp_all_reduce.attn",
            comm_group,
            sharding_formulas["fwd_tp_all_reduce_attn"].subs(SUBSTITUTE_VALUES),
            [cp_all_gather_sync_node.id],
            degree=TP
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
        all_reduce_ffn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.fwd.tp_all_reduce.ffn",
            comm_group,
            sharding_formulas["fwd_tp_all_reduce_ffn"].subs(SUBSTITUTE_VALUES),
            [tp_all_reduce_attn_sync_node.id],
            degree=TP
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
    DP=dp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)

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
        all_reduce_ffn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.bwd.tp_all_reduce.ffn",
            comm_group,
            sharding_formulas["bwd_tp_all_reduce_ffn"].subs(SUBSTITUTE_VALUES),
            [pre_layer_sync_node.id],
            degree=TP
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
        reduce_scatter_k = make_comm_node(
            "REDUCE_SCATTER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.cp_reduce_scatter.k",
            comm_group,
            sharding_formulas["bwd_cp_reduce_scatter_k"].subs(SUBSTITUTE_VALUES),
            [tp_all_reduce_ffn_sync_node.id],
            degree=CP
        )
        reduce_scatter_v = make_comm_node(
            "REDUCE_SCATTER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.cp_reduce_scatter.v",
            comm_group,
            sharding_formulas["bwd_cp_reduce_scatter_v"].subs(SUBSTITUTE_VALUES),
            [tp_all_reduce_ffn_sync_node.id],
            degree=CP
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

        all_reduce_attn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.bwd.tp_all_reduce.attn",
            comm_group,
            sharding_formulas["bwd_tp_all_reduce_attn"].subs(SUBSTITUTE_VALUES),
            [cp_reduce_scatter_sync_node.id],
            degree=TP
        )
        tp_all_reduce_attn_nodes.append(all_reduce_attn)

    tp_all_reduce_attn_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.post_tp_all_reduce.attn",
        [],
        0,
        [node.id for node in tp_all_reduce_attn_nodes]
    )

    #If this is the last microbatch, then current gradients are dp,cp sharded
    #I'm assuming FSDP is only across DP. So we will need to do reduce_scatter across DP followed by all_reduce across CP

    num_microbatches = B.subs(SUBSTITUTE_VALUES) // (M.subs(SUBSTITUTE_VALUES)*dp.subs(SUBSTITUTE_VALUES))
    grad_dp_reduce_scatter_nodes = []
    grad_cp_all_reduce_nodes = []
    if microbatch_id == num_microbatches - 1:

        grad_dp_reduce_scatter_comm_groups = permute_axes(-2,-1,pipeline_stage_from_layer_id(layer_id,L.subs(SUBSTITUTE_VALUES)),-1)
        for comm_group in grad_dp_reduce_scatter_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(grad_dp_reduce_scatter_comm_groups))]:
            grad_reduce_scatter = make_comm_node(
                "REDUCE_SCATTER",
                f"mb{microbatch_id}.layer{layer_id}.bwd.grad_dp_reduce_scatter",
                comm_group,
                sharding_formulas["bwd_fsdp_reduce_scatter"].subs(SUBSTITUTE_VALUES),
                [tp_all_reduce_attn_sync_node.id],
                degree=DP
            )
            grad_dp_reduce_scatter_nodes.append(grad_reduce_scatter)

        grad_dp_reduce_scatter_sync_node = Node(
            "SYNC",
            "BARRIER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.post_grad_dp_reduce_scatter",
            [],
            0,
            [node.id for node in grad_dp_reduce_scatter_nodes]
        )

        grad_cp_all_reduce_comm_groups = permute_axes(-1,-2,pipeline_stage_from_layer_id(layer_id,L.subs(SUBSTITUTE_VALUES)),-1)
        for comm_group in grad_cp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(grad_cp_all_reduce_comm_groups))]:
            grad_all_reduce = make_comm_node(
                "ALL_REDUCE",
                f"mb{microbatch_id}.layer{layer_id}.bwd.grad_cp_all_reduce",
                comm_group,
                sharding_formulas["bwd_cp_all_reduce"].subs(SUBSTITUTE_VALUES),
                [grad_dp_reduce_scatter_sync_node.id],
                degree=CP
            )
            grad_cp_all_reduce_nodes.append(grad_all_reduce)    

    post_layer_sync_node = Node(
        "SYNC",
        "POST_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.post_layer_sync",
        [],
        0,
        [node.id for node in grad_cp_all_reduce_nodes] if microbatch_id == num_microbatches - 1 else [tp_all_reduce_attn_sync_node.id]
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

    fwd_stages:Dict[int,Tuple[Node, Node]] = dict()
    for i in range(num_pipeline_stages):
        layer_ids = list(range(i*layers_per_stage, (i+1)*layers_per_stage))
        fwd_stages[i] = forward_pipeline_stage(layer_ids,i,microbatch_id)

    #Wire them up with SEND/RECV nodes
    for i in range(1,num_pipeline_stages):
        # Create SEND/RECV nodes. They will sit between the post_pipeline_sync of the previous stage and the pre_pipeline_sync of the current stage.
        pipeline_transfer_comm_groups = get_pp_transfer_comm_groups(i-1,i)
        fwd_send_recv_nodes = []
        for comm_group in pipeline_transfer_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(pipeline_transfer_comm_groups))]:
            send_recv_node = Node(
                "COMM",
                "SEND_RECV",
                f"mb{microbatch_id}.pipeline_stage{i-1}_to_{i}.fwd.send_recv",
                comm_group,
                sharding_formulas["pipeline_transfer"].subs(SUBSTITUTE_VALUES),
                [fwd_stages[i-1][1].id]  # Depends on the post_pipeline_sync of the previous stage
            )
            fwd_send_recv_nodes.append(send_recv_node)
        fwd_stages[i][0].deps.extend([node.id for node in fwd_send_recv_nodes])  # The pre_pipeline_sync of the current stage depends on the SEND/RECV nodes

    bwd_stages:Dict[int,Tuple[Node, Node]] = dict()
    for i in range(num_pipeline_stages):
        layer_ids = list(reversed(range((num_pipeline_stages-i-1)*layers_per_stage, (num_pipeline_stages-i)*layers_per_stage)))
        print(f"Layer_ids:{layer_ids}")
        bwd_stages[num_pipeline_stages-i-1] = backward_pipeline_stage(layer_ids,num_pipeline_stages-i-1,microbatch_id)

    #Wire them up
    for i in range(1,num_pipeline_stages):
        # Create SEND/RECV nodes. They will sit between the post_pipeline_sync of the previous stage and the pre_pipeline_sync of the current stage.
        pipeline_transfer_comm_groups = get_pp_transfer_comm_groups(i,i-1)
        bwd_send_recv_nodes = []
        for comm_group in pipeline_transfer_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(pipeline_transfer_comm_groups))]:
            send_recv_node = Node(
                "COMM",
                "SEND_RECV",
                f"mb{microbatch_id}.pipeline_stage{i}_to_{i-1}.bwd.send_recv",
                comm_group,
                sharding_formulas["pipeline_transfer"].subs(SUBSTITUTE_VALUES),
                [bwd_stages[i][1].id]  # Depends on the post_pipeline_sync of the current stage
            )
            bwd_send_recv_nodes.append(send_recv_node)
        bwd_stages[i-1][0].deps.extend([node.id for node in bwd_send_recv_nodes])

    # Connect forward and backward passes
    # This doesn't require a SEND_RECV because its the same ranks
    bwd_stages[num_pipeline_stages-1][0].deps.append(fwd_stages[num_pipeline_stages-1][1].id)

    return fwd_stages, bwd_stages

def get_pp_1f1b_schedule(num_pipeline_stages:int, num_microbatches:int):

    class Dir(enum.Enum):
        F = 1
        B = 2

    @dataclass
    class PPNode:
        dir:Dir
        microbatch:int

        def __str__(self):
            return f"{self.dir.name}{self.microbatch}"

    @dataclass
    class MBStatus:
        pp:int
        dir:Dir

        def __str__(self):
            return f"{self.pp},{self.dir.name}"

    @dataclass
    class PPRank:
        rank:int
        occupied:bool
        node:PPNode

        def __str__(self):
            if self.occupied:
                return str(self.node)
            else:
                return ""

    #Create hardware nodes for each PP rank
    ranks:Dict[int,PPRank] = {i:PPRank(rank=i, occupied=False, node=PPNode(dir=Dir.F, microbatch=-1)) for i in range(num_pipeline_stages)}

    # for i in range(num_pipeline_stages):
    #     print(f"Rank {i}: {ranks[i]}")

    #Maintain one list of to complete operation for each microbatch
    microbatch_ops:Dict[int,MBStatus] = {i:MBStatus(pp=0, dir=Dir.F) for i in range(num_microbatches)}

    # for i in range(num_microbatches):
    #     print(f"Microbatch {i}: {microbatch_ops[i]}")

    tick = 0

    steps:List[Dict[int,str]] = []

    #Number of forwards issued at stage i that haven't yet had their backward issued at
    #stage i. 1F1B's defining property is capping this at (num_pipeline_stages - i), so
    #a stage is only allowed to race ahead on new forwards up to that bound instead of
    #greedily issuing every forward that happens to be ready.
    in_flight:Dict[int,int] = {i:0 for i in range(num_pipeline_stages)}

    while True:

        if len(microbatch_ops) == 0:
            break

        #Reset ranks
        for i in range(num_pipeline_stages):
            ranks[i].occupied = False

        #Current step
        curr_step:Dict[int,str] = {pp_stage_id:"" for pp_stage_id,pp_stage in ranks.items()}

        def issue(mb_id:int, mb:MBStatus):
            ranks[mb.pp].occupied = True
            ranks[mb.pp].node = PPNode(dir=mb.dir, microbatch=mb_id)
            curr_step[mb.pp] = str(ranks[mb.pp].node)
            #Update the microbatch status to the next operation
            if mb.dir == Dir.F:
                if mb.pp == num_pipeline_stages - 1:
                    #Last stage, next op is backward
                    microbatch_ops[mb_id] = MBStatus(pp=mb.pp, dir=Dir.B)
                else:
                    #Next op is forward on next stage
                    microbatch_ops[mb_id] = MBStatus(pp=mb.pp + 1, dir=Dir.F)
            else:
                #Retiring this backward frees up this stage's in-flight slot.
                in_flight[mb.pp] -= 1
                if mb.pp == 0:
                    #Last stage, next op is done
                    del microbatch_ops[mb_id]
                else:
                    #Next op is backward on previous stage
                    microbatch_ops[mb_id] = MBStatus(pp=mb.pp - 1, dir=Dir.B)

        #Backward ops are issued first: they're never capped, and draining them before
        #considering any forward is what lets a capped-out stage make progress again in
        #the same tick a backward arrives, instead of losing a tick to ordering.
        for mb_id,mb in sorted(microbatch_ops.items()):
            if mb.dir != Dir.B or ranks[mb.pp].occupied:
                continue
            issue(mb_id, mb)

        #Forward ops only get issued up to the per-stage in-flight cap; beyond that the
        #rank is deliberately left idle (a bubble) even though the microbatch is ready,
        #which is exactly what bounds 1F1B's memory usage relative to GPipe.
        for mb_id,mb in sorted(microbatch_ops.items()):
            if mb.dir != Dir.F or ranks[mb.pp].occupied:
                continue
            if in_flight[mb.pp] >= num_pipeline_stages - mb.pp:
                continue
            in_flight[mb.pp] += 1
            issue(mb_id, mb)

        tick += 1

        # print(f"Tick {tick}")
        # for i in range(num_pipeline_stages):
        #         print(f"Rank {i}: {ranks[i]}")
        steps.append(curr_step)

    # print("Steps:")
    # for i, step in enumerate(steps):
    #     print(f"Tick {i}: {step}")

    display_steps:Dict[int,List[str]] = {}
    for step in steps:
        for pp_stage_id,op in step.items():
            if pp_stage_id not in display_steps:
                display_steps[pp_stage_id] = []
            display_steps[pp_stage_id].append(op)

    # for display_pp_stage_id,display_ops in display_steps.items():
    #     print(f"PP Stage {display_pp_stage_id}:",end='')
    #     op_list = [f"{op:^4}" for op in display_ops]
    #     print(" | ".join(op_list))

    dependency_chains:Dict[int,List[str]] = dict()
    for display_pp_stage_id,display_ops in display_steps.items():
        dependency_chains[display_pp_stage_id] = []
        for op in display_ops:
            if op != "":
                dependency_chains[display_pp_stage_id].append(op)

    return dependency_chains

def construct_1f1b_schedule(num_microbatches:int):
    #Get the dependency chains for each PP stage
    dependency_chains = get_pp_1f1b_schedule(pp.subs(SUBSTITUTE_VALUES), num_microbatches)

    #Generate the pipeline pass for each microbatch
    microbatch_passes:Dict[int,Tuple[Dict[int,Tuple[Node, Node]], Dict[int,Tuple[Node, Node]]]] = {}
    for microbatch_id in range(num_microbatches):
        microbatch_passes[microbatch_id] = pipeline_pass_for_single_microbatch(microbatch_id, L.subs(SUBSTITUTE_VALUES))

    #Wire up according to microbatch dependency chains
    for pp_stage_id,chain in dependency_chains.items():
        for i in range(1, len(chain)):
            prev_op = chain[i-1]
            curr_op = chain[i]
            prev_microbatch_id = int(prev_op[1:])
            curr_microbatch_id = int(curr_op[1:])
            prev_dir = prev_op[0]
            curr_dir = curr_op[0]

            if prev_microbatch_id == curr_microbatch_id:
                #This is a dependency within the same microbatch, which is already wired up in pipeline_pass_for_single_microbatch
                continue

            if prev_dir == "F" and curr_dir == "F":
                #Connect the post_pipeline_sync of the previous microbatch to the pre_pipeline_sync of the current microbatch
                microbatch_passes[curr_microbatch_id][0][pp_stage_id][0].deps.append(microbatch_passes[prev_microbatch_id][0][pp_stage_id][1].id)
            elif prev_dir == "B" and curr_dir == "B":
                #Connect the post_pipeline_sync of the previous microbatch to the pre_pipeline_sync of the current microbatch
                microbatch_passes[curr_microbatch_id][1][pp_stage_id][0].deps.append(microbatch_passes[prev_microbatch_id][1][pp_stage_id][1].id)
            elif prev_dir == "F" and curr_dir == "B":
                #Connect the post_pipeline_sync of the previous microbatch to the pre_pipeline_sync of the current microbatch
                microbatch_passes[curr_microbatch_id][1][pp_stage_id][0].deps.append(microbatch_passes[prev_microbatch_id][0][pp_stage_id][1].id)
            elif prev_dir == "B" and curr_dir == "F":
                #Connect the post_pipeline_sync of the previous microbatch to the pre_pipeline_sync of the current microbatch
                microbatch_passes[curr_microbatch_id][0][pp_stage_id][0].deps.append(microbatch_passes[prev_microbatch_id][1][pp_stage_id][1].id)
            else:
                raise ValueError(f"Invalid dependency chain: {prev_op} -> {curr_op}")

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

#Divisibility checks for model and training config
def check_divisibility():
    DP=dp.subs(SUBSTITUTE_VALUES)
    CP=cp.subs(SUBSTITUTE_VALUES)
    PP=pp.subs(SUBSTITUTE_VALUES)
    TP=tp.subs(SUBSTITUTE_VALUES)
    microbatch_size=M.subs(SUBSTITUTE_VALUES)
    global_batch_size=B.subs(SUBSTITUTE_VALUES)
    num_layers=L.subs(SUBSTITUTE_VALUES)

    #Batch size should be divisible by DP
    assert global_batch_size % DP == 0, f"Batch size {global_batch_size} is not divisible by data parallelism {DP}"
    #Batch size // DP should be divisible by microbatches
    assert (global_batch_size // DP) % microbatch_size == 0, f"Batch size {global_batch_size} // data parallelism {DP} = {global_batch_size//DP} is not divisible by microbatches {microbatch_size}"
    #Number of layers should be divisible by PP
    assert num_layers % PP == 0, f"Number of layers {num_layers} is not divisible by pipeline parallelism {PP}"
    #H_k should be divisible by TP
    assert H_k.subs(SUBSTITUTE_VALUES) % TP == 0, f"Attention head size {H_k.subs(SUBSTITUTE_VALUES)} is not divisible by tensor parallelism {TP}"
    #D_ff should be divisible by TP
    assert D_ff.subs(SUBSTITUTE_VALUES) % TP == 0, f"Feedforward dimension {D_ff.subs(SUBSTITUTE_VALUES)} is not divisible by tensor parallelism {TP}"
    #S should be divisible by CP
    assert S.subs(SUBSTITUTE_VALUES) % CP == 0, f"Sequence length {S.subs(SUBSTITUTE_VALUES)} is not divisible by sequence parallelism {CP}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a analytical model compatible DAG from model and training config")
    parser.add_argument("--single-comm", "-s", action="store_true", help="Generate a single comm group for all communication operations")

    #I'm fixing the ordering for sharding axes based on how deep inside the model they are located
    #dp replicates entire model
    #cp is at sequence level
    #pp happens at multi layer level
    #tp and ep happen at sublayer level
    args = parser.parse_args()
    if args.single_comm:
        MAX_COMM_GROUPS_PER_COMM = 1
    else:
        MAX_COMM_GROUPS_PER_COMM = 1000000

    sharding_axes_symbols = [dp,cp,pp,tp]
    sharding_axes = [s.subs(SUBSTITUTE_VALUES) for s in sharding_axes_symbols]

    check_divisibility()

    # fwd_layers = forward_pipeline_stage([0,1,2,3],0,0)
    # bwd_layers = backward_pipeline_stage([3,2,1,0],0,0)

    # # print(bwd_layers[0][0].name)
    # # print(fwd_layers[1][1].name)
    # bwd_layers[0][0].deps.append(fwd_layers[-1][1].id)

    # pipeline_pass_for_single_microbatch(0,6)

    # all_nodes_single_layer_single_microbatch = Node._all_nodes

    # write_trace_to_json(list(all_nodes_single_layer_single_microbatch.values()), "trace_single_layer.json")


    construct_1f1b_schedule(B.subs(SUBSTITUTE_VALUES) // (M.subs(SUBSTITUTE_VALUES)*dp.subs(SUBSTITUTE_VALUES)))

    write_trace_to_json(list(Node._all_nodes.values()), "trace_llama3.json")

    print((per_layer_weight_sizes()*tp*(L-2)+2*V*D).subs(SUBSTITUTE_VALUES))
    # per_layer_hand_caclulated = 2*D*H_k*D_h+D*H_q*D_h+D*D+3*D*D_ff
    # print(per_layer_hand_caclulated.subs(SUBSTITUTE_VALUES))
    # print(per_layer_hand_caclulated)