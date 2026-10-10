import argparse
import functools
import os
import sys
import json
import time
import sympy
from dataclasses import dataclass, field
from typing import List, Dict, Any, ClassVar, Set, Tuple
from itertools import product
import numpy as np
import enum

from tracegen import mem_footprint
from tracegen import compress

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

SUBSTITUTE_VALUES = dict()
# SUBSTITUTE_VALUES = {
#     L: 128,
#     H_k: 8,
#     H_q: 128,
#     D_h: 128,
#     D_ff: 53248,
#     V: 128000,
#     dp: 4,
#     pp: 16,
#     cp: 1,
#     tp: 8,
#     B: 32*4,
#     S: 8192,
#     M: 2
# }

# DP/PP/CP/TP/etc. below and RESOLVED_SHARDING_FORMULAS further down are
# resolved by resolve_constants(), called once SUBSTITUTE_VALUES has actually
# been populated (see __main__) -- SUBSTITUTE_VALUES starts empty now that
# it's filled in from CLI args rather than hardcoded here, so resolving them
# eagerly at module-load time, before argparse has even run, doesn't work
# (there's nothing to substitute yet).

MAX_COMM_GROUPS_PER_COMM = 1
#Bytes per element for MatMul.tensor_size_numeric -- 2 (bf16/fp16) is the
#de facto default training precision; overridden from --bytes-per-element.
BYTES_PER_ELEMENT = 2
#Whether backward can assume forward's fsdp_all_gather left weights resident
#-- overridden from --reuse-fwd-fsdp; False (the flag's own default) means
#backward re-fetches them itself on its first microbatch on each stage.
REUSE_FWD_FSDP = False
#Whether activations are recomputed in the backward pass (dropped after
#forward, regenerated via single_layer_recompute right before backward
#needs them -- the standard "activation checkpointing" memory/compute
#tradeoff) rather than kept resident the whole fwd/bwd gap -- overridden
#from --recompute-activations; False (the flag's own default) means
#activations stay resident and no recompute nodes get added.
RECOMPUTE_ACTIVATIONS = False

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
    comps:List = field(default_factory=list)

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

def make_comp_node(subtype:str, name:str, deps:List, comps:List)->Node:
    return Node(
        "COMP",
        subtype,
        name,
        [],
        0,
        deps,
        comps
    )

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

#Memoizes (str(expr), int(expr.subs(SUBSTITUTE_VALUES))) by expression. A
#full trace constructs thousands of MatMuls, but there are only a few dozen
#distinct FLOPs/tensor_size formulas among them -- the same shapes recur
#identically on every layer and every microbatch -- so without this, both
#sympy's .subs() (real symbolic-tree substitution, not a dict lookup) and its
#str() printer (also real tree-walking work) get redone thousands of times
#over for answers that were already computed. This was the dominant cost in
#trace generation once per-matmul comps were added.
_MATMUL_EXPR_CACHE:Dict[sympy.Expr, Tuple[str,int]] = {}

def _resolve_expr(expr:sympy.Expr) -> Tuple[str,int]:
    if expr not in _MATMUL_EXPR_CACHE:
        _MATMUL_EXPR_CACHE[expr] = (str(expr), int(expr.subs(SUBSTITUTE_VALUES)))
    return _MATMUL_EXPR_CACHE[expr]

@dataclass
class MatMul:
    name:str
    inputA:List[sympy.Expr]
    inputB:List[sympy.Expr]
    contracting_dims:List[sympy.Expr]
    output:List[sympy.Expr]
    FLOPs:sympy.Expr = field(init=False)
    FLOPs_str:str = field(init=False)
    FLOPs_numeric:int = field(init=False)
    tensor_size:sympy.Expr = field(init=False)
    tensor_size_str:str = field(init=False)
    tensor_size_numeric:int = field(init=False)

    def __post_init__(self):
        #Every contracting dim must actually be shared by both operands -- an
        #easy typo to make by hand, and one that would silently skew FLOPs.
        assert set(self.contracting_dims) <= set(self.inputA) & set(self.inputB), \
            f"{self.name}: contracting_dims {self.contracting_dims} not present in both inputA and inputB"
        #2 * (product of every dim appearing in either operand): each
        #contracting dim appears in both inputA and inputB but only once in
        #their union, so this is exactly 2 * output_size * contracted_size --
        #one multiply and one add per MAC -- without needing output at all.
        self.FLOPs = 2 * sympy.prod(set(self.inputA) | set(self.inputB))
        self.FLOPs_str, self.FLOPs_numeric = _resolve_expr(self.FLOPs)
        #Total element count moved for this op -- inputA + inputB + output,
        #each its own product (not a union: unlike FLOPs, a dim shared between
        #operands still occupies real memory in both, so it's counted twice).
        #Stays in elements, precision-agnostic, like a shape; the numeric
        #version below is where a concrete dtype width gets applied, for the
        #roofline model's memory-bound term (bytes_moved / bandwidth).
        self.tensor_size = sympy.prod(self.inputA) + sympy.prod(self.inputB) + sympy.prod(self.output)
        tensor_size_str, tensor_size_numeric = _resolve_expr(self.tensor_size)
        self.tensor_size_str = tensor_size_str
        self.tensor_size_numeric = tensor_size_numeric * BYTES_PER_ELEMENT

    def __str__(self):
        return f"[{','.join(str(x) for x in self.inputA)}] x [{','.join(str(x) for x in self.inputB)}] -> [{','.join(str(x) for x in self.output)}]"

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

def resolve_constants():
    """(Re)computes the plain-int module constants (DP/PP/CP/TP/...) and
    RESOLVED_SHARDING_FORMULAS from the current SUBSTITUTE_VALUES. Must be
    called once SUBSTITUTE_VALUES is fully populated -- from CLI args, see
    __main__ -- and before any trace generation, since every generation
    function reads these as globals rather than resolving them itself: doing
    the sympy .subs() substitution once here instead of inside every function
    (node_id_from_axes alone used to do it tens of thousands of times per
    run) is what keeps generation fast. See git history if you need the
    reasoning spelled out further.
    """
    global DP, PP, CP, TP, NUM_LAYERS, GLOBAL_BATCH_SIZE, MICROBATCH_SIZE, NUM_MICROBATCHES, RESOLVED_SHARDING_FORMULAS
    DP = int(dp.subs(SUBSTITUTE_VALUES))
    PP = int(pp.subs(SUBSTITUTE_VALUES))
    CP = int(cp.subs(SUBSTITUTE_VALUES))
    TP = int(tp.subs(SUBSTITUTE_VALUES))
    NUM_LAYERS = int(L.subs(SUBSTITUTE_VALUES))
    GLOBAL_BATCH_SIZE = int(B.subs(SUBSTITUTE_VALUES))
    MICROBATCH_SIZE = int(M.subs(SUBSTITUTE_VALUES))
    NUM_MICROBATCHES = GLOBAL_BATCH_SIZE // (MICROBATCH_SIZE * DP)
    RESOLVED_SHARDING_FORMULAS = {key: formula.subs(SUBSTITUTE_VALUES) * BYTES_PER_ELEMENT for key, formula in sharding_formulas.items()}



def pipeline_stage_from_layer_id(layer_id:int,num_layers:int) -> int:
    layers_per_stage = num_layers // PP
    return layer_id // layers_per_stage

# def node_id_from_axes(d,c=0,p=0,t=0):
def node_id_from_axes(dims):
    #All inputs are lists or arrays
    d = dims[0]
    c = dims[1]
    p = dims[2]
    t = dims[3]
    return d*CP*PP*TP + c*PP*TP + p*TP + t

def axes_from_node_id(node_id:int):
    d = node_id // (CP*PP*TP)
    c = (node_id % (PP*CP*TP)) // (PP*TP)
    p = (node_id % (CP*TP)) // (TP)
    t = node_id % TP

    return d,p,c,t

def get_pp_transfer_comm_groups(from_stage:int,to_stage:int):
    comm_group_coords = []

    for d in range(DP):
        for c in range(CP):
            for t in range(TP):
                comm_group_coords.append([(d,c,from_stage,t),(d,c,to_stage,t)])
    
    comm_group_node_ids = [[node_id_from_axes(coord) for coord in group] for group in comm_group_coords]
    return comm_group_node_ids


def permute_axes(d:int,c:int,p:int,t:int)->List[List[int]]:
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

def single_layer_forward_pass(layer_id,microbatch_id,prefetch_dep_node_id=None):
    #FSDP fetch is outside the scope of this function

    pipeline_stage_id = pipeline_stage_from_layer_id(layer_id,NUM_LAYERS)

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
        #Prefetch: gather THIS layer's weights while the PREVIOUS layer is still
        #computing, instead of waiting for this layer's own pre_layer_sync --
        #bounded to exactly one layer ahead, matching llm_dense_full_pass.tex's
        #PF_{p,m} pseudocode (gathers l+1's weights concurrently with F_{l,m}).
        #The first layer of a stage has no earlier layer to overlap with, so
        #the caller leaves prefetch_dep_node_id unset and it falls back to
        #this layer's own pre_layer_sync, same as before this change.
        gather_dep_id = prefetch_dep_node_id if prefetch_dep_node_id is not None else pre_layer_sync_node.id
        for comm_group in fsdp_all_gather_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(fsdp_all_gather_comm_groups))]:
            all_gather = make_comm_node(
                "ALL_GATHER",
                f"mb{microbatch_id}.layer{layer_id}.fwd.fsdp_all_gather",
                comm_group,
                RESOLVED_SHARDING_FORMULAS["fwd_fsdp_all_gather"],
                [gather_dep_id],
                degree=DP
            )
            fsdp_all_gather_nodes.append(all_gather)

    #For mb==0, fsdp_all_gather_nodes alone isn't enough: since the prefetch
    #change, they depend on the PREVIOUS layer's start (for overlap), not this
    #layer's own turn -- so without pre_layer_sync_node here too, this layer's
    #compute would only wait for its weights to arrive, not for the previous
    #layer (same physical NPU) to actually finish. pre_layer_sync_node already
    #transitively requires that (see the caller's per-layer chaining), so
    #adding it back here is what makes this layer's compute wait for both
    #"weights ready" and "NPU free" instead of just the former.
    post_fsdp_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.fwd.post_fsdp_all_gather",
        [],
        0,
        ([node.id for node in fsdp_all_gather_nodes] + [pre_layer_sync_node.id]) if microbatch_id == 0 else [pre_layer_sync_node.id]
    )

    kvq_projections = [[
        MatMul("X@W_k",[M,S/cp,D],[D,H_k/tp,D_h],[D],[M,S/cp,H_k/tp,D_h]),
        MatMul("X@W_v",[M,S/cp,D],[D,H_k/tp,D_h],[D],[M,S/cp,H_k/tp,D_h]),
        MatMul("X@W_q",[M,S/cp,D],[D,H_q/tp,D_h],[D],[M,S/cp,H_q/tp,D_h])
    ]]

    comp_kvq_projection_node = make_comp_node(
        "MATMUL",
        f"mb{microbatch_id}.layer{layer_id}.fwd.kvq_projections",
        [post_fsdp_sync_node.id],
        kvq_projections
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
            RESOLVED_SHARDING_FORMULAS["fwd_cp_all_gather_k"],
            [comp_kvq_projection_node.id],
            degree=CP
        )
        all_gather_v = make_comm_node(
            "ALL_GATHER",
            f"mb{microbatch_id}.layer{layer_id}.fwd.cp_all_gather.v",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["fwd_cp_all_gather_v"],
            [comp_kvq_projection_node.id],
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

    comp_attn_score_output_projection = [
        [MatMul("Q@K",[M,S/cp,H_q/tp,D_h],[M,S,H_k/tp,D_h],[D_h],[M,H_q/tp,S/cp,S])],
        [MatMul("S@V",[M,H_q/tp,S/cp,S],[M,S,H_k/tp,D_h],[S],[M,S/cp,H_q/tp,D_h])],
        [MatMul("A@W_oA",[M,S/cp,H_q/tp,D_h],[H_q/tp,D_h,D],[H_q/tp,D_h],[M,S/cp,D])]
    ]

    comp_attn_score_output_projection_node = make_comp_node(
        "MATMUL",
        f"mb{microbatch_id}.layer{layer_id}.fwd.attn_score_output_projection",
        [cp_all_gather_sync_node.id],
        comp_attn_score_output_projection
    )

    tp_all_reduce_comm_groups = permute_axes(-1,-1, pipeline_stage_id,-2)
    tp_all_reduce_attn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:
        all_reduce_attn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.fwd.tp_all_reduce.attn",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["fwd_tp_all_reduce_attn"],
            [comp_attn_score_output_projection_node.id],
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

    comp_ffn = [
        [
            MatMul("X@W_1",[M,S/cp,D],[D,D_ff/tp],[D],[M,S/cp,D_ff/tp]),
            MatMul("X@W_2",[M,S/cp,D],[D,D_ff/tp],[D],[M,S/cp,D_ff/tp]),
        ],
        [MatMul("X@W_oF",[M,S/cp,D_ff/tp],[D_ff/tp,D],[D_ff/tp],[M,S/cp,D])]
    ]
    comp_ffn_node = make_comp_node(
        "MATMUL",
        f"mb{microbatch_id}.layer{layer_id}.fwd.ffn",
        [tp_all_reduce_attn_sync_node.id],
        comp_ffn
    )
    
    tp_all_reduce_ffn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:
        all_reduce_ffn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.fwd.tp_all_reduce.ffn",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["fwd_tp_all_reduce_ffn"],
            [comp_ffn_node.id],
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

def single_layer_recompute(layer_id, microbatch_id, pipeline_stage_id, segment, dep_id):
    """Regenerates ONE of the two groups of forward tensors activation
    checkpointing dropped, as a fresh second occurrence of just the forward
    ops that produce it -- not a full copy of forward's compute, since
    backward doesn't need everything forward produced:

    segment="ffn": regenerates F1/F2 (comp_ffn's X@W_1, X@W_2 matmuls) --
    what comp_ffn_backward reads. Excludes the down-projection (X@W_oF ->
    F_o); comp_ffn_backward never reads F_o.

    segment="attn": regenerates K/V/Q/K_hat/V_hat/S/A (kvq_projection, a
    REAL cp_all_gather -- regenerating K_hat/V_hat means re-doing the actual
    CP network transfer, not just local compute -- and the score/attn
    matmuls) -- what comp_attn_backward reads. Excludes the output
    projection (A@W_oA -> A_o); comp_attn_backward never reads A_o.

    dep_id is the single dependency this segment chains onto. The caller is
    responsible for making it whatever this layer's backward has reached by
    the point this segment's own first consumer needs it (ffn: right after
    this layer's backward starts; attn: after the ffn-backward chain,
    matching comp_attn_backward's own current entry point) -- this segment
    runs on the same physical NPU as the rest of this layer's backward, so
    it must be ordered into that SAME single-NPU sequential chain rather
    than independently rooted in a way the simulator could treat as
    runnable concurrently with the rest of backward.

    Returns the last node's id, for the caller to point its real backward
    matmul's dep at instead of depending on dep_id directly.
    """
    if segment == "ffn":
        recompute_ffn = [[
            MatMul("X@W_1",[M,S/cp,D],[D,D_ff/tp],[D],[M,S/cp,D_ff/tp]),
            MatMul("X@W_2",[M,S/cp,D],[D,D_ff/tp],[D],[M,S/cp,D_ff/tp]),
        ]]
        recompute_ffn_node = make_comp_node(
            "MATMUL",
            f"mb{microbatch_id}.layer{layer_id}.bwd.recompute_ffn",
            [dep_id],
            recompute_ffn
        )
        return recompute_ffn_node.id

    elif segment == "attn":
        recompute_kvq = [[
            MatMul("X@W_k",[M,S/cp,D],[D,H_k/tp,D_h],[D],[M,S/cp,H_k/tp,D_h]),
            MatMul("X@W_v",[M,S/cp,D],[D,H_k/tp,D_h],[D],[M,S/cp,H_k/tp,D_h]),
            MatMul("X@W_q",[M,S/cp,D],[D,H_q/tp,D_h],[D],[M,S/cp,H_q/tp,D_h])
        ]]
        recompute_kvq_node = make_comp_node(
            "MATMUL",
            f"mb{microbatch_id}.layer{layer_id}.bwd.recompute_kvq_projection",
            [dep_id],
            recompute_kvq
        )

        cp_all_gather_comm_groups = permute_axes(-1,-2,pipeline_stage_id,-1)
        recompute_cp_all_gather_nodes = []
        for comm_group in cp_all_gather_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(cp_all_gather_comm_groups))]:
            all_gather_k = make_comm_node(
                "ALL_GATHER",
                f"mb{microbatch_id}.layer{layer_id}.bwd.recompute_cp_all_gather.k",
                comm_group,
                RESOLVED_SHARDING_FORMULAS["fwd_cp_all_gather_k"],
                [recompute_kvq_node.id],
                degree=CP
            )
            all_gather_v = make_comm_node(
                "ALL_GATHER",
                f"mb{microbatch_id}.layer{layer_id}.bwd.recompute_cp_all_gather.v",
                comm_group,
                RESOLVED_SHARDING_FORMULAS["fwd_cp_all_gather_v"],
                [recompute_kvq_node.id],
                degree=CP
            )
            recompute_cp_all_gather_nodes.append(all_gather_k)
            recompute_cp_all_gather_nodes.append(all_gather_v)

        recompute_cp_all_gather_sync_node = Node(
            "SYNC",
            "BARRIER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.recompute_post_cp_all_gather",
            [],
            0,
            [node.id for node in recompute_cp_all_gather_nodes]
        )

        recompute_score_attn = [
            [MatMul("Q@K",[M,S/cp,H_q/tp,D_h],[M,S,H_k/tp,D_h],[D_h],[M,H_q/tp,S/cp,S])],
            [MatMul("S@V",[M,H_q/tp,S/cp,S],[M,S,H_k/tp,D_h],[S],[M,S/cp,H_q/tp,D_h])],
        ]
        recompute_attn_node = make_comp_node(
            "MATMUL",
            f"mb{microbatch_id}.layer{layer_id}.bwd.recompute_attn_score",
            [recompute_cp_all_gather_sync_node.id],
            recompute_score_attn
        )
        return recompute_attn_node.id

    else:
        raise ValueError(f"Unknown recompute segment: {segment!r}")

def single_layer_backward_pass(layer_id,microbatch_id):
    #FSDP reduce scatter is outside the scope of this function
    pipeline_stage_id = pipeline_stage_from_layer_id(layer_id,NUM_LAYERS)

    pre_layer_sync_node = Node(
        "SYNC",
        "PRE_LAYER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.pre_layer_sync",
        [],
        0,
        []
    )

    #If weights aren't assumed to still be resident from the forward pass
    #(REUSE_FWD_FSDP=False), the first microbatch to reach backward on this
    #stage needs to re-fetch them before any backward matmul can run -- same
    #mb_id==0 gate as forward's own fetch, since 1F1B preserves microbatch
    #ordering: mb 0 is always the first to reach both forward AND backward on
    #any given stage (see get_pp_1f1b_schedule).
    bwd_fsdp_all_gather_nodes = []
    if not REUSE_FWD_FSDP and microbatch_id == 0:
        bwd_fsdp_all_gather_comm_groups = permute_axes(-2,-1,pipeline_stage_id,-1)
        for comm_group in bwd_fsdp_all_gather_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(bwd_fsdp_all_gather_comm_groups))]:
            all_gather = make_comm_node(
                "ALL_GATHER",
                f"mb{microbatch_id}.layer{layer_id}.bwd.fsdp_all_gather",
                comm_group,
                RESOLVED_SHARDING_FORMULAS["fwd_fsdp_all_gather"],
                [pre_layer_sync_node.id],
                degree=DP
            )
            bwd_fsdp_all_gather_nodes.append(all_gather)

    post_fsdp_sync_node = Node(
        "SYNC",
        "BARRIER",
        f"mb{microbatch_id}.layer{layer_id}.bwd.post_fsdp_all_gather",
        [],
        0,
        ([node.id for node in bwd_fsdp_all_gather_nodes] + [pre_layer_sync_node.id]) if bwd_fsdp_all_gather_nodes else [pre_layer_sync_node.id]
    )

    # dF_o (=dLoss) is the layer's backward input. Stage 1 (dF, dW_oF) both
    # depend only on dF_o -- parallel. The elementwise dF_1=dF*F_2, dF_2=dF*F_1
    # (not matmuls, not modeled) gate stage 2: dW_1/dW_2 (weight grads) and the
    # two dRA_partial terms all depend only on dF_1/dF_2, which are both ready
    # at the same time -- so all four are parallel too, even though dW_1/dW_2
    # don't feed anything downstream (they're GradSync outputs) while the
    # dRA_partial terms do (their sum, also unmodeled, feeds tp_all_reduce.ffn).
    comp_ffn_backward = [
        [
            MatMul("dFo@WoF",[M,S/cp,D],[D_ff/tp,D],[D],[M,S/cp,D_ff/tp]),
            MatMul("F@dFo",[M,S/cp,D_ff/tp],[M,S/cp,D],[M,S/cp],[D_ff/tp,D]),
        ],
        [
            MatMul("RA@dF1",[M,S/cp,D],[M,S/cp,D_ff/tp],[M,S/cp],[D,D_ff/tp]),
            MatMul("RA@dF2",[M,S/cp,D],[M,S/cp,D_ff/tp],[M,S/cp],[D,D_ff/tp]),
            MatMul("dF1@W1",[M,S/cp,D_ff/tp],[D,D_ff/tp],[D_ff/tp],[M,S/cp,D]),
            MatMul("dF2@W2",[M,S/cp,D_ff/tp],[D,D_ff/tp],[D_ff/tp],[M,S/cp,D]),
        ]
    ]
    ffn_backward_dep_id = post_fsdp_sync_node.id
    if RECOMPUTE_ACTIVATIONS:
        ffn_backward_dep_id = single_layer_recompute(layer_id, microbatch_id, pipeline_stage_id, "ffn", post_fsdp_sync_node.id)

    comp_ffn_backward_node = make_comp_node(
        "MATMUL",
        f"mb{microbatch_id}.layer{layer_id}.bwd.ffn",
        [ffn_backward_dep_id],
        comp_ffn_backward
    )

    tp_all_reduce_comm_groups = permute_axes(-1,-1, pipeline_stage_from_layer_id(layer_id,NUM_LAYERS),-2)
    tp_all_reduce_ffn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:
        all_reduce_ffn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.bwd.tp_all_reduce.ffn",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["bwd_tp_all_reduce_ffn"],
            [comp_ffn_backward_node.id],
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

    # dA_o (=dRA, post tp_all_reduce.ffn + the unmodeled residual add) is this
    # block's input. Stage 1 (dA, dW_oA) both depend only on dA_o -- parallel.
    # Stage 2 (dS, dV_hat) both depend only on dA (from stage 1) -- parallel;
    # dW_oA doesn't feed either, it's a GradSync output. Stage 3 (dQ, dK_hat)
    # both depend only on dS (from stage 2) -- parallel; dV_hat doesn't feed
    # either, it heads straight to cp_reduce_scatter.v.
    comp_attn_backward = [
        [
            MatMul("dAo@WoA",[M,S/cp,D],[H_q/tp,D_h,D],[D],[M,S/cp,H_q/tp,D_h]),
            MatMul("A@dAo",[M,S/cp,H_q/tp,D_h],[M,S/cp,D],[M,S/cp],[H_q/tp,D_h,D]),
        ],
        [
            MatMul("dA@Vhat",[M,S/cp,H_q/tp,D_h],[M,S,H_k/tp,D_h],[D_h],[M,H_q/tp,S/cp,S]),
            MatMul("S@dA",[M,H_q/tp,S/cp,S],[M,S/cp,H_q/tp,D_h],[S/cp],[M,S,H_k/tp,D_h]),
        ],
        [
            MatMul("dS@Khat",[M,H_q/tp,S/cp,S],[M,S,H_k/tp,D_h],[S],[M,S/cp,H_q/tp,D_h]),
            MatMul("dS@Q",[M,H_q/tp,S/cp,S],[M,S/cp,H_q/tp,D_h],[S/cp],[M,S,H_k/tp,D_h]),
        ]
    ]
    attn_backward_dep_id = tp_all_reduce_ffn_sync_node.id
    if RECOMPUTE_ACTIVATIONS:
        attn_backward_dep_id = single_layer_recompute(layer_id, microbatch_id, pipeline_stage_id, "attn", tp_all_reduce_ffn_sync_node.id)

    comp_attn_backward_node = make_comp_node(
        "MATMUL",
        f"mb{microbatch_id}.layer{layer_id}.bwd.attn_score_output_projection",
        [attn_backward_dep_id],
        comp_attn_backward
    )

    cp_reduce_scatter_comm_groups = permute_axes(-1,-2,pipeline_stage_from_layer_id(layer_id,NUM_LAYERS),-1)
    cp_reduce_scatter_nodes = []
    for comm_group in cp_reduce_scatter_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(cp_reduce_scatter_comm_groups))]:
        reduce_scatter_k = make_comm_node(
            "REDUCE_SCATTER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.cp_reduce_scatter.k",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["bwd_cp_reduce_scatter_k"],
            [comp_attn_backward_node.id],
            degree=CP
        )
        reduce_scatter_v = make_comm_node(
            "REDUCE_SCATTER",
            f"mb{microbatch_id}.layer{layer_id}.bwd.cp_reduce_scatter.v",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["bwd_cp_reduce_scatter_v"],
            [comp_attn_backward_node.id],
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

    # dK, dV (post cp_reduce_scatter) and dQ (transitively ready too, via
    # comp_attn_backward as an ancestor of cp_reduce_scatter_sync_node) are
    # all available here, so every matmul below -- the three weight grads and
    # the three dX_partial terms -- depends only on them plus pre-existing
    # X/weights. All six are independent of each other -- one parallel group.
    comp_kvq_backward = [[
        MatMul("X@dK",[M,S/cp,D],[M,S/cp,H_k/tp,D_h],[M,S/cp],[D,H_k/tp,D_h]),
        MatMul("X@dV",[M,S/cp,D],[M,S/cp,H_k/tp,D_h],[M,S/cp],[D,H_k/tp,D_h]),
        MatMul("X@dQ",[M,S/cp,D],[M,S/cp,H_q/tp,D_h],[M,S/cp],[D,H_q/tp,D_h]),
        MatMul("dK@Wk",[M,S/cp,H_k/tp,D_h],[D,H_k/tp,D_h],[H_k/tp,D_h],[M,S/cp,D]),
        MatMul("dV@Wv",[M,S/cp,H_k/tp,D_h],[D,H_k/tp,D_h],[H_k/tp,D_h],[M,S/cp,D]),
        MatMul("dQ@Wq",[M,S/cp,H_q/tp,D_h],[D,H_q/tp,D_h],[H_q/tp,D_h],[M,S/cp,D]),
    ]]
    comp_kvq_backward_node = make_comp_node(
        "MATMUL",
        f"mb{microbatch_id}.layer{layer_id}.bwd.kvq_projections",
        [cp_reduce_scatter_sync_node.id],
        comp_kvq_backward
    )

    #Reuse same comm  groups as before
    tp_all_reduce_attn_nodes = []
    for comm_group in tp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(tp_all_reduce_comm_groups))]:

        all_reduce_attn = make_comm_node(
            "ALL_REDUCE",
            f"mb{microbatch_id}.layer{layer_id}.bwd.tp_all_reduce.attn",
            comm_group,
            RESOLVED_SHARDING_FORMULAS["bwd_tp_all_reduce_attn"],
            [comp_kvq_backward_node.id],
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

    num_microbatches = NUM_MICROBATCHES
    grad_dp_reduce_scatter_nodes = []
    grad_cp_all_reduce_nodes = []
    if microbatch_id == num_microbatches - 1:

        grad_dp_reduce_scatter_comm_groups = permute_axes(-2,-1,pipeline_stage_from_layer_id(layer_id,NUM_LAYERS),-1)
        for comm_group in grad_dp_reduce_scatter_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(grad_dp_reduce_scatter_comm_groups))]:
            grad_reduce_scatter = make_comm_node(
                "REDUCE_SCATTER",
                f"mb{microbatch_id}.layer{layer_id}.bwd.grad_dp_reduce_scatter",
                comm_group,
                RESOLVED_SHARDING_FORMULAS["bwd_fsdp_reduce_scatter"],
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

        grad_cp_all_reduce_comm_groups = permute_axes(-1,-2,pipeline_stage_from_layer_id(layer_id,NUM_LAYERS),-1)
        for comm_group in grad_cp_all_reduce_comm_groups[:min(MAX_COMM_GROUPS_PER_COMM, len(grad_cp_all_reduce_comm_groups))]:
            grad_all_reduce = make_comm_node(
                "ALL_REDUCE",
                f"mb{microbatch_id}.layer{layer_id}.bwd.grad_cp_all_reduce",
                comm_group,
                RESOLVED_SHARDING_FORMULAS["bwd_cp_all_reduce"],
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
    # print(f"Received layer_ids for forward pass({pipeline_stage_id}): {layer_ids}")
    layers = []
    L = len(layer_ids)
    prefetch_dep_node_id = None
    for layer_id in layer_ids:
        layer_nodes = single_layer_forward_pass(layer_id,microbatch_id,prefetch_dep_node_id)
        layers.append(layer_nodes)
        #Next layer's fsdp_all_gather prefetches against THIS layer's start.
        prefetch_dep_node_id = layer_nodes[0].id

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
        # print(f"L;{layer_id}/{layer_id%L} -> L-1;{layer_id-1}/{layer_id%L-1}")
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
    # print(f"Received layer_ids for backward pass({pipeline_stage_id}): {layer_ids}")
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
        # print(f"L;{layer_id+1}/{layer_id%L+1} -> L;{layer_id}/{layer_id%L}")
        # print(f"Connecting layer {layer_id%L+1}'s post_layer_sync to layer {layer_id%L}'s pre_layer_sync")
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
    print(f"Generating pipeline pass for microbatch {microbatch_id}")
    num_pipeline_stages = PP
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
                RESOLVED_SHARDING_FORMULAS["pipeline_transfer"],
                [fwd_stages[i-1][1].id]  # Depends on the post_pipeline_sync of the previous stage
            )
            fwd_send_recv_nodes.append(send_recv_node)
        fwd_stages[i][0].deps.extend([node.id for node in fwd_send_recv_nodes])  # The pre_pipeline_sync of the current stage depends on the SEND/RECV nodes

    bwd_stages:Dict[int,Tuple[Node, Node]] = dict()
    for i in range(num_pipeline_stages):
        layer_ids = list(reversed(range((num_pipeline_stages-i-1)*layers_per_stage, (num_pipeline_stages-i)*layers_per_stage)))
        # print(f"Layer_ids:{layer_ids}")
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
                RESOLVED_SHARDING_FORMULAS["pipeline_transfer"],
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
    dependency_chains = get_pp_1f1b_schedule(PP, num_microbatches)

    #Generate the pipeline pass for each microbatch
    microbatch_passes:Dict[int,Tuple[Dict[int,Tuple[Node, Node]], Dict[int,Tuple[Node, Node]]]] = {}
    for microbatch_id in range(num_microbatches):
        microbatch_passes[microbatch_id] = pipeline_pass_for_single_microbatch(microbatch_id, NUM_LAYERS)

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

#Same reasoning as _MATMUL_EXPR_CACHE: individual dims (e.g. "S/cp") recur
#identically across thousands of matmul instances, so str() on each one is
#worth memoizing too rather than re-printing the same handful of dims over
#and over.
@functools.lru_cache(maxsize=None)
def _dim_str(dim:sympy.Expr) -> str:
    return str(dim)

def matmul_to_json(matmul:MatMul) -> Dict:
    return {
        "name": matmul.name,
        "inputA": [_dim_str(dim) for dim in matmul.inputA],
        "inputB": [_dim_str(dim) for dim in matmul.inputB],
        "contracting_dims": [_dim_str(dim) for dim in matmul.contracting_dims],
        "output": [_dim_str(dim) for dim in matmul.output],
        "FLOPs": matmul.FLOPs_str,
        "FLOPs_numeric": matmul.FLOPs_numeric,
        "tensor_size": matmul.tensor_size_str,
        "tensor_size_numeric": matmul.tensor_size_numeric,
    }

def write_trace_to_json(nodes:List[Node], filepath:str, compressed:bool=True) -> None:
    #Ignoring comm group deduplication (CommGroupTracker) for now — comm_group is written
    #as the literal list of involved NPU ids per node, for visual inspection.
    trace = [
        {
            "id": node.id,
            "type": node.type,
            "subtype": node.subtype,
            "name": node.name,
            "comm_group": [int(npu_id) for npu_id in sorted(node.comm_group)],
            "size": int(node.size),
            "deps": node.deps,
            "comps": [[matmul_to_json(matmul) for matmul in stage] for stage in node.comps],
        }
        for node in sorted(nodes, key=lambda n: n.id)
    ]
    json_obj = {
        "nodes": trace,
    }
    #Writes zstd-compressed directly by default -- the uncompressed form
    #(tens of GB for this project's larger configs) is never written to disk
    #at all. See compress.py's module docstring for the json.dumps()-vs-
    #json.dump() and compression-level reasoning; run compress.py directly
    #(`python src/trace/compress.py decompress ...`) to materialize a plain
    #.json for manual inspection, or pass compressed=False here to skip
    #compression entirely (e.g. for a small trace you want to read directly).
    if compressed:
        compress.write_compressed_trace(json_obj, filepath)
    else:
        with open(filepath, "w") as f:
            json.dump(json_obj, f)

def report_memory_footprint() -> None:
    """Computes and prints the peak per-rank memory footprint (worst-case
    pipeline stage -- stage 0, see mem_footprint.peak_rank_memory's
    docstring) using mem_footprint.py's standalone liveness model, plus which
    single step within a layer's forward/backward it's likely to occur in.

    reuse_fwd_weights=REUSE_FWD_FSDP mirrors --reuse-fwd-fsdp directly, and
    activation_recompute=RECOMPUTE_ACTIVATIONS mirrors --recompute-activations
    directly -- both this module's and mem_footprint.py's conventions now
    agree on what True means for each ("stays resident, no re-fetch/re-gather
    needed" and "dropped after forward, regenerated via a recompute segment"
    respectively), so both pass straight through with no inversion needed.
    """
    mem_footprint.SUBSTITUTE_VALUES.clear()
    mem_footprint.SUBSTITUTE_VALUES.update(SUBSTITUTE_VALUES)

    layer_repr = mem_footprint.LayerReprForFoorprint(activation_recompute=RECOMPUTE_ACTIVATIONS, reuse_fwd_weights=REUSE_FWD_FSDP)
    result = layer_repr.peak_rank_memory(
        pp_degree=PP,
        dp_degree=DP,
        layers_per_stage=NUM_LAYERS // PP,
        bytes_per_element=BYTES_PER_ELEMENT,
    )

    BYTES_TO_GB = 2**30

    working_set_bytes = result["total_peak"] * BYTES_PER_ELEMENT
    optimizer_bytes = result["optimizer_state_bytes_per_rank"]
    print()
    print("=== Peak memory footprint (worst-case rank, pipeline stage 0) ===")
    print(f"Assumptions: reuse_fwd_weights={REUSE_FWD_FSDP}, recompute_activations={RECOMPUTE_ACTIVATIONS}, {BYTES_PER_ELEMENT} bytes/element, dp={DP}")
    print(f"total_peak_bytes: {result['total_peak_bytes'] / BYTES_TO_GB:.2f} GB "
          f"(working set {result['total_peak']:,} elements = {working_set_bytes / BYTES_TO_GB:.2f} GB "
          f"+ optimizer state {optimizer_bytes / BYTES_TO_GB:.2f} GB)")
    print(f"likely step: '{result['peak_step_name']}' ({result['peak_step_phase']} pass)")
    print(f"breakdown:\npersistent_per_layer={result['persistent_per_layer']:,}\n"
          f"fwd_peak_per_layer={result['fwd_peak_per_layer']:,}\n"
          f"bwd_peak_per_layer={result['bwd_peak_per_layer']:,}")
    print(f"active_microbatch_peak={result['active_microbatch_peak']:,}\n"
          f"other_in_flight_contribution={result['other_in_flight_contribution']:,}")


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
    parser.add_argument("--layers", "-l", type=int, default=128, help="Number of layers in the model")
    parser.add_argument("--kv-heads",type=int,default=8,help="Number of key/value heads in the model")
    parser.add_argument("--q-heads",type=int,default=128,help="Number of query heads in the model")
    parser.add_argument("--head-dim",type=int,default=128,help="Dimension of each attention head")
    parser.add_argument("--ffn-dim",type=int,default=53248,help="Feedforward dimension in the model")
    parser.add_argument("--vocab",type=int,default=128000,help="Vocabulary size")
    parser.add_argument("--seq-len",type=int,default=8192,help="Sequence length")
    parser.add_argument("--batch-size",type=int,default=32*4,help="Global batch size")
    parser.add_argument("--microbatch-size",type=int,default=2,help="Microbatch size")
    parser.add_argument("--dp",type=int,default=4,help="Data parallelism degree")
    parser.add_argument("--cp",type=int,default=1,help="Sequence parallelism degree")
    parser.add_argument("--pp",type=int,default=4,help="Pipeline parallelism degree")
    parser.add_argument("--tp",type=int,default=8,help="Tensor parallelism degree")
    parser.add_argument("--output","-o",type=str,default="trace.json.zst",help="Output file path for the generated trace (written zstd-compressed; a .zst suffix is appended if missing)")
    parser.add_argument("--bytes-per-element",type=int,default=2,help="Bytes per element for MatMul tensor sizes (default: 2, bf16/fp16)")
    parser.add_argument(
        "--reuse-fwd-fsdp",
        action="store_true",
        help="Assume the FSDP-gathered weights from the forward pass are still resident when backward "
             "runs. Default: false -- backward's first microbatch on each stage re-fetches them via an "
             "extra fsdp_all_gather. Pass --reuse-fwd-fsdp to assume they're still resident instead "
             "(the original fsdp_all_gather-on-mb0-forward-only pattern, no extra backward gather)."
    )
    parser.add_argument(
        "--recompute-activations",
        action="store_true",
        help="Recompute activations in the backward pass instead of keeping them resident -- the standard "
             "'activation checkpointing' memory/compute tradeoff. Default: false -- activations stay "
             "resident from forward through backward, nothing dropped or regenerated. Pass "
             "--recompute-activations to instead drop F1/F2 and K/V/Q/K_hat/V_hat/S/A after forward and "
             "regenerate them via extra recompute nodes (single_layer_recompute) right before backward "
             "needs them."
    )
    parser.add_argument("-m","--mem-only",action="store_true",help="Use this flag if you just want to know estimated memory footprint only and aren't interested in trace generation")

    #I'm fixing the ordering for sharding axes based on how deep inside the model they are located
    #dp replicates entire model
    #cp is at sequence level
    #pp happens at multi layer level
    #tp and ep happen at sublayer level
    args = parser.parse_args()

    SUBSTITUTE_VALUES = {
        L: args.layers,
        H_k: args.kv_heads,
        H_q: args.q_heads,
        D_h: args.head_dim,
        D_ff: args.ffn_dim,
        V: args.vocab,
        S: args.seq_len,
        B: args.batch_size,
        M: args.microbatch_size,
        dp: args.dp,
        cp: args.cp,
        pp: args.pp,
        tp: args.tp
    }

    if args.single_comm:
        MAX_COMM_GROUPS_PER_COMM = 1
    else:
        MAX_COMM_GROUPS_PER_COMM = 1000000

    #Must be set before resolve_constants(), which bakes the current
    #BYTES_PER_ELEMENT into RESOLVED_SHARDING_FORMULAS (comm node sizes) --
    #resolving first and overriding this after would silently keep comm
    #sizes at the stale module-level default regardless of --bytes-per-element.
    BYTES_PER_ELEMENT = args.bytes_per_element
    REUSE_FWD_FSDP = args.reuse_fwd_fsdp
    RECOMPUTE_ACTIVATIONS = args.recompute_activations

    resolve_constants()

    sharding_axes_symbols = [dp,cp,pp,tp]
    sharding_axes = [s.subs(SUBSTITUTE_VALUES) for s in sharding_axes_symbols]

    check_divisibility()

    print(f"NUM NPUS: {(tp*pp*cp*dp).subs(SUBSTITUTE_VALUES)}")

    # fwd_layers = forward_pipeline_stage([0,1,2,3],0,0)
    # bwd_layers = backward_pipeline_stage([3,2,1,0],0,0)

    # # print(bwd_layers[0][0].name)
    # # print(fwd_layers[1][1].name)
    # bwd_layers[0][0].deps.append(fwd_layers[-1][1].id)

    # pipeline_pass_for_single_microbatch(0,6)

    # all_nodes_single_layer_single_microbatch = Node._all_nodes

    # write_trace_to_json(list(all_nodes_single_layer_single_microbatch.values()), "trace_single_layer.json")

    if not args.mem_only:
        gen_start_time = time.time()
        construct_1f1b_schedule(NUM_MICROBATCHES)
        gen_end_time = time.time()
        print(f"Generated trace in {gen_end_time - gen_start_time:.2f}s")

        output_path = args.output if compress.is_compressed(args.output) else args.output + compress.COMPRESSED_SUFFIX
        wr_start_time = time.time()
        write_trace_to_json(list(Node._all_nodes.values()), output_path)
        wr_end_time = time.time()
        print(f"Wrote trace to {output_path} in {wr_end_time - wr_start_time:.2f}s")

    report_memory_footprint()

    print((per_layer_weight_sizes()*tp*(L-2)+2*V*D).subs(SUBSTITUTE_VALUES))
    # per_layer_hand_caclulated = 2*D*H_k*D_h+D*H_q*D_h+D*D+3*D*D_ff
    # print(per_layer_hand_caclulated.subs(SUBSTITUTE_VALUES))
    # print(per_layer_hand_caclulated)