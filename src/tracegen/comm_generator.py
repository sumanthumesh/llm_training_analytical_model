import os
from tracegen.trace_generator import make_comm_node, Node, write_trace_to_json
from tracegen import compress
import coords
import argparse

TP = 0
PP = 0
CP = 0
DP = 0

AXIS_ORDER = coords.DEFAULT_AXIS_ORDER

degree_size = {}

def _sizes():
    return {"dp": DP, "cp": CP, "pp": PP, "tp": TP}

def coords_to_rank(dp,cp,pp,tp):
    return coords.coords_to_rank({"dp": dp, "cp": cp, "pp": pp, "tp": tp}, _sizes(), AXIS_ORDER)

def rank_to_coords(rank):
    c = coords.rank_to_coords(rank, _sizes(), AXIS_ORDER)
    return c["dp"], c["cp"], c["pp"], c["tp"]

def get_comm_group(axis:str):
    if axis == "dp":
        return [[coords_to_rank(dp,cp,pp,tp) for dp in range(DP)] for cp in range(CP) for pp in range(PP) for tp in range(TP)]
    elif axis == "cp":
        return [[coords_to_rank(dp,cp,pp,tp) for cp in range(CP)] for dp in range(DP) for pp in range(PP) for tp in range(TP)]
    elif axis == "pp":
        return [[coords_to_rank(dp,cp,pp,tp) for pp in range(PP)] for dp in range(DP) for cp in range(CP) for tp in range(TP)]
    elif axis == "tp":
        return [[coords_to_rank(dp,cp,pp,tp) for tp in range(TP)] for dp in range(DP) for cp in range(CP) for pp in range(PP)]
    else:
        raise ValueError(f"Invalid axis: {axis}")

def main():
    parser = argparse.ArgumentParser(description="Generate communication nodes to run on analytical model.")
    parser.add_argument("--parallelism","-p",nargs=4,type=int,help="Parallelism config of the network in the order (dp,cp,pp,tp)")
    parser.add_argument(
        "--axis-order",
        type=coords.axis_order_arg,
        default=coords.DEFAULT_AXIS_ORDER,
        help="Outermost-to-innermost nesting of dp,cp,pp,tp into one global NPU rank, comma-separated "
             f"(default: {','.join(coords.DEFAULT_AXIS_ORDER)}). Must match whatever order trace_generator.py "
             "and the topology were generated with."
    )
    parser.add_argument("--type","-t",type=str,choices=["ALL_GATHER","ALL_REDUCE","REDUCE_SCATTER"],help="Type of communication node to generate")
    parser.add_argument("--axis","-a",type=str,choices=["dp","cp","pp","tp"],help="Axis of the communication node to generate")
    parser.add_argument("--size","-s",type=float,help="Size of the communication in GB")
    parser.add_argument("--output","-o",type=str,help="Output file path to save the generated communication node")
    parser.add_argument("--compressed",action="store_true",help="Write a plain .json trace instead of zstd-compressed .json.zst")
    parser.add_argument("--single",action="store_true",help="Emit only rank 0's comm group instead of one per group across the whole topology.")
    args = parser.parse_args()

    global TP, PP, CP, DP, AXIS_ORDER
    TP = args.parallelism[3]
    PP = args.parallelism[2]
    CP = args.parallelism[1]
    DP = args.parallelism[0]
    AXIS_ORDER = args.axis_order

    global degree_size
    degree_size = {
        "dp": DP,
        "cp": CP,
        "pp": PP,
        "tp": TP
    }

    comm_group = get_comm_group(args.axis)
    if args.single:
        comm_group = comm_group[:1]  # first group returned always contains rank 0
    print(comm_group)
    for cg in comm_group:
        comm_node = make_comm_node(args.type,
                               f"{args.axis}_{args.type.lower()}_{args.size:.2f}GB", 
                               cg, 
                               args.size*2**30,
                               [],
                               degree=degree_size[args.axis])

    if args.compressed:
        output_path = args.output if compress.is_compressed(args.output) else args.output + compress.COMPRESSED_SUFFIX
    else:
        output_path = args.output
    write_trace_to_json(list(Node._all_nodes.values()), output_path, compressed=args.compressed, axis_order=AXIS_ORDER)

if __name__ == "__main__":
    main()