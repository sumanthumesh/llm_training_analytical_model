import sympy
from typing import List, Dict, Set
from dataclasses import dataclass, field

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

@dataclass
class Tensor:
    name: str
    shape: List[sympy.Expr]
    size: sympy.Expr = field(init=False)

    def __post_init__(self):
        self.size = sympy.prod(self.shape)

class LayerReprForFoorprint:
    def __init__(self,activation_recompute:bool,reuse_fwd_weights:bool):
        self.activation_recompute = activation_recompute
        self.reuse_fwd_weights = reuse_fwd_weights
        self.weights = self.all_weights()
        self.fwd_tensors = self.fwd_activations()
        self.bwd_tensors, self.bwd_weight_grads = self.bwd_gradients()

    def all_weights(self):
        weights:Dict[str, Tensor] = dict()
        weights["W_k"] = Tensor("W_k",[D,H_k/tp,D_h])
        weights["W_v"] = Tensor("W_v",[D,H_k/tp,D_h])
        weights["W_q"] = Tensor("W_q",[D,H_q/tp,D_h])
        weights["W_oA"] = Tensor("W_oA",[H_q/tp,D_h,D])
        weights["W1"] = Tensor("W1",[D,D_ff/tp])
        weights["W2"] = Tensor("W2",[D,D_ff/tp])
        weights["W_oF"] = Tensor("W_oF",[D_ff/tp,D])
        return weights

    def fwd_activations(self):
        activations:Dict[str, Tensor] = dict()
        activations["X"] = Tensor("X",[M,S/cp,D])

        activations["K"] = Tensor("K",[M,S/cp,H_k/tp,D_h])
        activations["V"] = Tensor("V",[M,S/cp,H_k/tp,D_h])
        activations["Q"] = Tensor("Q",[M,S/cp,H_q/tp,D_h])

        activations["K_hat"] = Tensor("K_hat",[M,S,H_k/tp,D_h])
        activations["V_hat"] = Tensor("V_hat",[M,S,H_k/tp,D_h])

        activations["S"] = Tensor("S",[M,H_q/tp,S/cp,S])
        activations["A"] = Tensor("A",[M,S/cp,H_q/tp,D_h])

        activations["A_o"] = Tensor("A_o",[M,S/cp,D])

        activations["A_o_hat"] = Tensor("A_o_hat",[M,S/cp,D])

        activations["RA"] = Tensor("RA",[M,S/cp,D])

        activations["F1"] = Tensor("F1",[M,S/cp,D_ff/tp])
        activations["F2"] = Tensor("F2",[M,S/cp,D_ff/tp])
        activations["F"] = Tensor("F",[M,S/cp,D_ff/tp])
        activations["F_o"] = Tensor("F_o",[M,S/cp,D])

        activations["F_o_hat"] = Tensor("F_o_hat",[M,S/cp,D])

        return activations

    @dataclass
    class Step:
        name:str
        inputs:List[Tensor]
        outputs:List[Tensor]
        footprint:List[Tensor] = field(default_factory=list)

    def fwd_steps(self):
        fwd_layer_steps:List[LayerReprForFoorprint.Step] = []
        fsdp_all_gather = LayerReprForFoorprint.Step(
            name="fsdp_all_gather",
            inputs=list(self.weights.values()),
            outputs=list(self.weights.values())
        )
        kvq_projection = LayerReprForFoorprint.Step(
            name="kvq_projection",
            inputs=[self.fwd_tensors["X"], self.weights["W_k"], self.weights["W_v"], self.weights["W_q"]],
            outputs=[self.fwd_tensors["K"], self.fwd_tensors["V"], self.fwd_tensors["Q"]]
        )
        cp_all_gather = LayerReprForFoorprint.Step(
            name="cp_all_gather",
            inputs=[self.fwd_tensors["K"], self.fwd_tensors["V"]],
            outputs=[self.fwd_tensors["K_hat"], self.fwd_tensors["V_hat"]]
        )
        score = LayerReprForFoorprint.Step(
            name="score",
            inputs=[self.fwd_tensors["K_hat"], self.fwd_tensors["Q"]],
            outputs=[self.fwd_tensors["S"]]
        )
        attn = LayerReprForFoorprint.Step(
            name="attn",
            inputs=[self.fwd_tensors["S"], self.fwd_tensors["V_hat"]],
            outputs=[self.fwd_tensors["A"]]
        )
        attn_out_proj = LayerReprForFoorprint.Step(
            name="attn_out_proj",
            inputs=[self.fwd_tensors["A"], self.weights["W_oA"]],
            outputs=[self.fwd_tensors["A_o"]]
        )
        tp_all_reduce_attn = LayerReprForFoorprint.Step(
            name="tp_all_reduce_attn",
            inputs=[self.fwd_tensors["A_o"]],
            outputs=[self.fwd_tensors["A_o_hat"]]
        )
        residual_add = LayerReprForFoorprint.Step(
            name="residual_add",
            inputs=[self.fwd_tensors["A_o_hat"], self.fwd_tensors["X"]],
            outputs=[self.fwd_tensors["RA"]]
        )
        ffn1 = LayerReprForFoorprint.Step(
            name="ffn1",
            inputs=[self.fwd_tensors["RA"], self.weights["W1"], self.weights["W2"]],
            outputs=[self.fwd_tensors["F1"], self.fwd_tensors["F2"]]
        )       
        ffn2 = LayerReprForFoorprint.Step(
            name="ffn2",
            inputs=[self.fwd_tensors["F1"], self.fwd_tensors["F2"]],
            outputs=[self.fwd_tensors["F"]]
        )
        ffn_out_proj = LayerReprForFoorprint.Step(
            name="ffn_out_proj",
            inputs=[self.fwd_tensors["F"], self.weights["W_oF"]],
            outputs=[self.fwd_tensors["F_o"]]
        )
        tp_all_reduce_ffn = LayerReprForFoorprint.Step(
            name="tp_all_reduce_ffn",
            inputs=[self.fwd_tensors["F_o"]],
            outputs=[self.fwd_tensors["F_o_hat"]]
        )
        fwd_layer_steps = [
            fsdp_all_gather,
            kvq_projection,
            cp_all_gather,
            score,
            attn,
            attn_out_proj,
            tp_all_reduce_attn,
            residual_add,
            ffn1,
            ffn2,
            ffn_out_proj,
            tp_all_reduce_ffn
        ]
        return fwd_layer_steps


    def bwd_gradients(self):
        
        tensor_gradients:Dict[str, Tensor] = dict()
        weight_gradients:Dict[str, Tensor] = dict()
        
        tensor_gradients["dLoss"] = Tensor("dLoss",[M,S/cp,D])

        tensor_gradients["dRA_residual"] = Tensor("dRA_residual",[M,S/cp,D])
        tensor_gradients["dF_o_hat"] = Tensor("dF_o_hat",[M,S/cp,D])
        
        tensor_gradients["dF_o"] = Tensor("dF_o",[M,S/cp,D])
        tensor_gradients["dF"] = Tensor("dF",[M,S/cp,D_ff/tp])
        weight_gradients["dW_oF"] = Tensor("dW_oF",[D_ff/tp,D])

        tensor_gradients["dF2"] = Tensor("dF2",[M,S/cp,D_ff/tp])
        tensor_gradients["dF1"] = Tensor("dF1",[M,S/cp,D_ff/tp])
        weight_gradients["dW2"] = Tensor("dW2",[D,D_ff/tp])
        weight_gradients["dW1"] = Tensor("dW1",[D,D_ff/tp])

        tensor_gradients["dRA_partial"] = Tensor("dRA_partial",[M,S/cp,D])

        tensor_gradients["dRA_FFN"] = Tensor("dRA_FFN",[M,S/cp,D])
        tensor_gradients["dRA"] = Tensor("dRA",[M,S/cp,D])

        tensor_gradients["dA_o_hat"] = Tensor("dA_o_hat",[M,S/cp,D])
        tensor_gradients["dX_residual"] = Tensor("dX_residual",[M,S/cp,D])

        tensor_gradients["dA_o"] = Tensor("dA_o",[M,S/cp,D])
        tensor_gradients["dA"] = Tensor("dA",[M,S/cp,H_q/tp,D_h])
        weight_gradients["dW_oA"] = Tensor("dW_oA",[H_q/tp,D_h,D])

        tensor_gradients["dS"] = Tensor("dS",[M,H_q/tp,S/cp,S])
        tensor_gradients["dV_hat"] = Tensor("dV_hat",[M,S,H_k/tp,D_h])
        tensor_gradients["dQ"] = Tensor("dQ",[M,S/cp,H_q/tp,D_h])
        tensor_gradients["dK_hat"] = Tensor("dK_hat",[M,S,H_k/tp,D_h])
        tensor_gradients["dK"] = Tensor("dK",[M,S/cp,H_k/tp,D_h])
        tensor_gradients["dV"] = Tensor("dV",[M,S/cp,H_k/tp,D_h])

        weight_gradients["dW_k"] = Tensor("dW_k",[D,H_k/tp,D_h])
        weight_gradients["dW_v"] = Tensor("dW_v",[D,H_k/tp,D_h])
        weight_gradients["dW_q"] = Tensor("dW_q",[D,H_q/tp,D_h])

        tensor_gradients["dX_partial"] = Tensor("dX_partial",[M,S/cp,D])

        tensor_gradients["dX_attn"] = Tensor("dX_attn",[M,S/cp,D])

        tensor_gradients["dX"] = Tensor("dX",[M,S/cp,D])

        return tensor_gradients, weight_gradients

    def bwd_steps(self):

        bwd_layer_steps:List[LayerReprForFoorprint.Step] = []

        bwd_residual_add = LayerReprForFoorprint.Step(
            name="bwd_residual_add",
            inputs=[self.bwd_tensors["dLoss"]], 
            outputs=[self.bwd_tensors["dRA_residual"], self.bwd_tensors["dF_o_hat"]]
        )

        # Backward of the forward tp_all_reduce on F_o: a sum's gradient flows
        # to each contributing tp-rank's local partial unchanged, so this is
        # an identity rename, not a real collective (see llm_dense_full_pass
        # Sec. "Single Layer, Single Microbatch, Backward Pass" -> "Gated
        # Feed Forward", dF_o <- dF_o_hat).
        bwd_ffn_output_identity = LayerReprForFoorprint.Step(
            name="bwd_ffn_output_identity",
            inputs=[self.bwd_tensors["dF_o_hat"]],
            outputs=[self.bwd_tensors["dF_o"]]
        )

        bwd_ffn_out_proj = LayerReprForFoorprint.Step(
            name="bwd_ffn_out_proj",
            inputs=[self.bwd_tensors["dF_o"], self.weights["W_oF"], self.fwd_tensors["F"]],
            outputs=[self.bwd_tensors["dF"], self.bwd_weight_grads["dW_oF"]]
        )

        # Elementwise, not a matmul (trace_generator.py doesn't model it for
        # timing either) -- still real dataflow for memory purposes: F =
        # F1 (elementwise*) F2, so dF1 <- dF*F2, dF2 <- dF*F1.
        bwd_ffn_hadamard = LayerReprForFoorprint.Step(
            name="bwd_ffn_hadamard",
            inputs=[self.bwd_tensors["dF"], self.fwd_tensors["F1"], self.fwd_tensors["F2"]],
            outputs=[self.bwd_tensors["dF1"], self.bwd_tensors["dF2"]]
        )

        bwd_ffn1 = LayerReprForFoorprint.Step(
            name="bwd_ffn1",
            inputs=[self.fwd_tensors["RA"], self.bwd_tensors["dF1"], self.bwd_tensors["dF2"], self.weights["W1"], self.weights["W2"]],
            outputs=[self.bwd_weight_grads["dW1"], self.bwd_weight_grads["dW2"], self.bwd_tensors["dRA_partial"]]
        )

        # This one IS a real all-reduce: dRA_partial is each tp-rank's own
        # partial sum over its D_ff/tp contraction shard, not yet the true
        # gradient.
        bwd_tp_all_reduce_ffn_dRA = LayerReprForFoorprint.Step(
            name="bwd_tp_all_reduce_ffn_dRA",
            inputs=[self.bwd_tensors["dRA_partial"]],
            outputs=[self.bwd_tensors["dRA_FFN"]]
        )

        bwd_add_residual_ffn = LayerReprForFoorprint.Step(
            name="bwd_add_residual_ffn",
            inputs=[self.bwd_tensors["dRA_FFN"], self.bwd_tensors["dRA_residual"]],
            outputs=[self.bwd_tensors["dRA"]]
        )

        bwd_residual_add2 = LayerReprForFoorprint.Step(
            name="bwd_residual_add2",
            inputs=[self.bwd_tensors["dRA"]],
            outputs=[self.bwd_tensors["dA_o_hat"], self.bwd_tensors["dX_residual"]]
        )

        # Identity, same reasoning as bwd_ffn_output_identity, for the
        # attention block's tp_all_reduce.
        bwd_attn_output_identity = LayerReprForFoorprint.Step(
            name="bwd_attn_output_identity",
            inputs=[self.bwd_tensors["dA_o_hat"]],
            outputs=[self.bwd_tensors["dA_o"]]
        )

        bwd_attn_out_proj = LayerReprForFoorprint.Step(
            name="bwd_attn_out_proj",
            inputs=[self.bwd_tensors["dA_o"], self.weights["W_oA"], self.fwd_tensors["A"]],
            outputs=[self.bwd_tensors["dA"], self.bwd_weight_grads["dW_oA"]]
        )

        bwd_attn_score_grad = LayerReprForFoorprint.Step(
            name="bwd_attn_score_grad",
            inputs=[self.bwd_tensors["dA"], self.fwd_tensors["V_hat"], self.fwd_tensors["S"]],
            outputs=[self.bwd_tensors["dS"], self.bwd_tensors["dV_hat"]]
        )

        bwd_attn_qk_grad = LayerReprForFoorprint.Step(
            name="bwd_attn_qk_grad",
            inputs=[self.bwd_tensors["dS"], self.fwd_tensors["K_hat"], self.fwd_tensors["Q"]],
            outputs=[self.bwd_tensors["dQ"], self.bwd_tensors["dK_hat"]]
        )

        # Backward dual of the forward cp_all_gather on K, V.
        bwd_cp_reduce_scatter = LayerReprForFoorprint.Step(
            name="bwd_cp_reduce_scatter",
            inputs=[self.bwd_tensors["dK_hat"], self.bwd_tensors["dV_hat"]],
            outputs=[self.bwd_tensors["dK"], self.bwd_tensors["dV"]]
        )

        bwd_kvq_projection = LayerReprForFoorprint.Step(
            name="bwd_kvq_projection",
            inputs=[self.fwd_tensors["X"], self.bwd_tensors["dK"], self.bwd_tensors["dV"], self.bwd_tensors["dQ"]],
            outputs=[self.bwd_weight_grads["dW_k"], self.bwd_weight_grads["dW_v"], self.bwd_weight_grads["dW_q"]]
        )

        bwd_kvq_dx_partial = LayerReprForFoorprint.Step(
            name="bwd_kvq_dx_partial",
            inputs=[self.bwd_tensors["dK"], self.bwd_tensors["dV"], self.bwd_tensors["dQ"], self.weights["W_k"], self.weights["W_v"], self.weights["W_q"]],
            outputs=[self.bwd_tensors["dX_partial"]]
        )

        # Real all-reduce again: dX_partial is each tp-rank's partial
        # contraction-shard sum (backward dual of replicating X across tp).
        bwd_tp_all_reduce_kvq_dX = LayerReprForFoorprint.Step(
            name="bwd_tp_all_reduce_kvq_dX",
            inputs=[self.bwd_tensors["dX_partial"]],
            outputs=[self.bwd_tensors["dX_attn"]]
        )

        bwd_add_residual_x = LayerReprForFoorprint.Step(
            name="bwd_add_residual_x",
            inputs=[self.bwd_tensors["dX_attn"], self.bwd_tensors["dX_residual"]],
            outputs=[self.bwd_tensors["dX"]]
        )

        bwd_layer_steps = [
            bwd_residual_add,
            bwd_ffn_output_identity,
            bwd_ffn_out_proj,
            bwd_ffn_hadamard,
            bwd_ffn1,
            bwd_tp_all_reduce_ffn_dRA,
            bwd_add_residual_ffn,
            bwd_residual_add2,
            bwd_attn_output_identity,
            bwd_attn_out_proj,
            bwd_attn_score_grad,
            bwd_attn_qk_grad,
            bwd_cp_reduce_scatter,
            bwd_kvq_projection,
            bwd_kvq_dx_partial,
            bwd_tp_all_reduce_kvq_dX,
            bwd_add_residual_x,
        ]
        return bwd_layer_steps

    def compute_liveness(self, steps:List["LayerReprForFoorprint.Step"], pinned_names:Set[str]=set()) -> None:
        """Live-variable analysis over a step sequence -- intended to be
        called as self.compute_liveness(self.fwd_steps() + self.bwd_steps()),
        forward followed by backward in actual execution order, so a tensor
        produced in forward and consumed in backward (e.g. RA, F, A, S, Q,
        K_hat, V_hat, X, or a weight reused by its backward matmul) is
        tracked across the whole gap instead of looking dead at the end of
        forward.

        Sets .footprint on every step in place: the tensors resident during
        that step, i.e. its own inputs and outputs, plus anything produced or
        needed earlier that's still required by some later step.

        A tensor that's an output somewhere but never an input anywhere in
        `steps` (a weight gradient, or dX handed to the previous pipeline
        stage) is treated as live through the end of the given sequence --
        there's no information here about when something outside this list
        (a grad-sync collective, the optimizer step) actually consumes it, so
        "stays resident" is the safer default rather than "dies the instant
        it's produced." Override by hand for anything you know dies earlier.

        pinned_names: tensor names that never get pruned once introduced --
        they stay in every subsequent step's footprint through the end of
        `steps`, regardless of whether anything in `steps` uses them again.
        Lets a caller model "this tensor stays resident for reasons outside
        what this step list shows" (see layer_liveness's reuse_fwd_weights).
        """
        last_use_index:Dict[str, int] = {}
        for i, step in enumerate(steps):
            for t in step.inputs:
                last_use_index[t.name] = i

        produced_names = {t.name for step in steps for t in step.outputs}
        last_step_index = len(steps) - 1
        for name in produced_names - set(last_use_index.keys()):
            last_use_index[name] = last_step_index
        for name in pinned_names:
            last_use_index[name] = last_step_index

        live:Dict[str, Tensor] = {}
        for i, step in enumerate(steps):
            for t in step.inputs:
                live[t.name] = t
            for t in step.outputs:
                live[t.name] = t

            step.footprint = list(live.values())

            live = {name:t for name,t in live.items() if last_use_index.get(name, last_step_index) > i}

    def _recompute_segment(self, step_names:List[str]) -> List["LayerReprForFoorprint.Step"]:
        """A fresh replay of the named forward steps (a fresh self.fwd_steps()
        call, not the same Step objects as the real forward pass -- reusing
        the same objects would let this second occurrence's .footprint
        overwrite the first's when compute_liveness assigns to both).
        Renamed with a "recompute." prefix so it's visibly distinct from the
        real forward occurrence when inspecting steps.
        """
        segment = [s for s in self.fwd_steps() if s.name in step_names]
        for s in segment:
            s.name = f"recompute.{s.name}"
        return segment

    def _splice_recompute(self, bwd:List["LayerReprForFoorprint.Step"]) -> List["LayerReprForFoorprint.Step"]:
        """Inserts recompute segments into bwd_steps() just before the first
        backward step that needs each one -- "just in time", matching
        backward's own FFN-then-attention order and minimizing how long the
        recomputed tensors have to stay resident (the point of checkpointing
        in the first place):

        - Recompute FFN from RA (ffn1, ffn2 -> F1, F2, F) right before
          bwd_ffn_out_proj, its first consumer.
        - Recompute attention from X (kvq_projection, cp_all_gather, score,
          attn -> K, V, Q, K_hat, V_hat, S, A) right before bwd_attn_out_proj,
          its first consumer.
        """
        recompute_ffn = self._recompute_segment(["ffn1", "ffn2"])
        recompute_attn = self._recompute_segment(["kvq_projection", "cp_all_gather", "score", "attn"])

        spliced = []
        for step in bwd:
            if step.name == "bwd_ffn_out_proj":
                spliced.extend(recompute_ffn)
            if step.name == "bwd_attn_out_proj":
                spliced.extend(recompute_attn)
            spliced.append(step)
        return spliced

    def layer_liveness(self) -> List["LayerReprForFoorprint.Step"]:
        """Returns fwd_steps() + bwd_steps() (with recompute segments spliced
        in if activation_recompute) with .footprint populated on every step,
        honoring both reuse_fwd_weights and activation_recompute.

        The base pass is always the single combined-sequence liveness (
        bridges any tensor used on both sides of the fwd/bwd gap by default).
        On top of that:

        - reuse_fwd_weights=True: weights are pinned -- once gathered, they
          stay resident through the whole sequence ("always keep weights in
          the footprint").
        - reuse_fwd_weights=False: weights are dropped across the fwd/bwd
          gap (see _drop_tensors_across_gap) -- die at their natural forward
          last-use, reappear fresh at their first backward use.
        - activation_recompute=True: every forward activation except X and RA
          (the saved checkpoints) is dropped across the gap the same way,
          but a recompute segment is spliced into backward first so each one
          has a real step producing it again before backward actually needs
          it, instead of just vanishing.
        """
        fwd = self.fwd_steps()
        bwd = self.bwd_steps()
        if self.activation_recompute:
            bwd = self._splice_recompute(bwd)
        steps = fwd + bwd
        fwd_len = len(fwd)

        pinned_names = set(self.weights.keys()) if self.reuse_fwd_weights else set()
        self.compute_liveness(steps, pinned_names=pinned_names)

        drop_names:Set[str] = set()
        if not self.reuse_fwd_weights:
            drop_names |= set(self.weights.keys())
        if self.activation_recompute:
            drop_names |= set(self.fwd_tensors.keys()) - {"X", "RA"}

        if drop_names:
            self._drop_tensors_across_gap(steps, fwd_len, drop_names)

        return steps

    def _drop_tensors_across_gap(self, steps:List["LayerReprForFoorprint.Step"], fwd_len:int, names:Set[str]) -> None:
        """Removes each name in `names` from every step's .footprint
        strictly between its last use in steps[:fwd_len] (forward) and its
        first appearance in steps[fwd_len:] (backward, including any spliced
        recompute segment) -- see layer_liveness. Only touches the given
        names; every other tensor's footprint (already computed by the
        combined-sequence compute_liveness pass) is left as-is.
        """
        for name in names:
            last_fwd_idx = max(
                (i for i in range(fwd_len) if name in {t.name for t in steps[i].inputs}),
                default=None
            )
            first_bwd_idx = min(
                (i for i in range(fwd_len, len(steps))
                 if name in {t.name for t in steps[i].inputs} | {t.name for t in steps[i].outputs}),
                default=None
            )
            if last_fwd_idx is None or first_bwd_idx is None:
                continue
            for i in range(last_fwd_idx + 1, first_bwd_idx):
                steps[i].footprint = [t for t in steps[i].footprint if t.name != name]

    def optimizer_state_bytes_per_rank(
        self,
        dp_degree:int,
        layers_per_stage:int,
        master_bytes_per_param:int=4,
        momentum_bytes_per_param:int=4,
        variance_bytes_per_param:int=4,
    ) -> int:
        """Per-rank bytes for the FSDP-sharded optimizer state: an fp32
        master weight copy plus Adam's fp32 momentum (exp_avg) and variance
        (exp_avg_sq), one triple per parameter, for every layer on this
        pipeline stage. Always fp32 (4+4+4=12 bytes/param by default) by
        construction -- independent of whatever precision the compute-path
        weights/activations/gradients use (that's bytes_per_element,
        elsewhere) -- so this is computed directly in bytes rather than
        elements, and is NOT part of peak_rank_memory's total_peak; it's a
        separate, always-resident contribution: unlike the gathered working
        copy (which appears/disappears across the fwd/bwd gap per
        reuse_fwd_weights), the sharded master+momentum+variance state never
        leaves memory for the life of the run. Sharded 1/dp_degree per rank,
        same as the working-copy weights.
        """
        per_layer_params = int(sum(t.size for t in self.weights.values()).subs(SUBSTITUTE_VALUES))
        total_params_this_stage = per_layer_params * layers_per_stage
        bytes_per_param = master_bytes_per_param + momentum_bytes_per_param + variance_bytes_per_param
        return (total_params_this_stage * bytes_per_param) // dp_degree

    def peak_rank_memory(self, pp_degree:int, dp_degree:int, layers_per_stage:int, bytes_per_element:int, in_flight:int=None) -> Dict[str, int]:
        """Peak single-rank (one pipeline stage) footprint, honoring this
        instance's activation_recompute / reuse_fwd_weights policy. Returns both
        an elements-denominated breakdown (persistent_per_layer etc., same
        convention as Tensor.size -- for comparing step-to-step shapes) and a
        final total_peak_bytes that folds in optimizer_state_bytes_per_rank,
        which is NOT expressible in elements (see optimizer_state_bytes_per_rank)
        and so can't be part of total_peak itself.

        Requires SUBSTITUTE_VALUES to already be populated with concrete
        values for every symbol Tensor shapes use (D_h, H_q, S, M, cp, tp,
        ...) -- picking the larger of two footprints needs an actual number,
        not an unresolved sympy expression (max() can't compare those).

        in_flight defaults to pp_degree -- stage 0's 1F1B in-flight bound,
        the worst case across stages; pass pp_degree - stage_id for a
        specific later stage.

        Derivation: within one microbatch's single-layer fwd/bwd, only the
        tensors still resident at the very end of forward -- the footprint
        of fwd_steps()'s last step, under this policy -- persist across
        OTHER layers on this stage and OTHER in-flight microbatches.
        Everything else in a layer's footprint is transient to whichever
        single layer/microbatch is actively computing right now, so it
        should NOT be multiplied by layers_per_stage or in_flight wholesale
        (that overcounts data that's already been freed by the time a later
        layer/microbatch runs):

        - (in_flight - 1) OTHER in-flight microbatches have already finished
          forward for every layer on this stage, each contributing
          layers_per_stage copies of persistent_per_layer.
        - the one ACTIVELY-computing microbatch contributes
          (layers_per_stage - 1) copies of persistent_per_layer (the other
          layers on this stage it has already passed through) plus whichever
          of its own forward or backward transient peak is larger -- checked
          empirically rather than assumed, since the .tex doc's "backward is
          worse" claim isn't ground truth here either.

        total_peak's peak moment occurs exactly when the actively-computing
        microbatch reaches its own highest-footprint step -- every other
        contribution (other in-flight microbatches, other already-passed
        layers on this stage) is a constant background present throughout, so
        the step achieving active_transient_peak IS where the global peak
        happens. "peak_step_name" names that step (forward step name, a bwd_*
        name, or a recompute.*-prefixed name if activation_recompute spliced a
        recompute segment in ahead of it); "peak_step_phase" says which half.

        total_peak_bytes = total_peak * bytes_per_element + optimizer_state_bytes_per_rank
        -- the former is the active working set (compute-path precision, the
        only thing that varies with bytes_per_element), the latter is the
        always-resident fp32 master+momentum+variance shard (see
        optimizer_state_bytes_per_rank) -- fixed regardless of
        bytes_per_element, and present even while this stage is otherwise
        completely idle.
        """
        if in_flight is None:
            in_flight = pp_degree

        steps = self.layer_liveness()
        fwd_len = len(self.fwd_steps())

        def footprint_size(footprint):
            return int(sum((t.size for t in footprint), sympy.Integer(0)).subs(SUBSTITUTE_VALUES))

        sized_steps = [(s.name, footprint_size(s.footprint)) for s in steps]
        persistent_per_layer = footprint_size(steps[fwd_len - 1].footprint)
        fwd_peak_name, fwd_peak_per_layer = max(sized_steps[:fwd_len], key=lambda ns: ns[1])
        bwd_peak_name, bwd_peak_per_layer = max(sized_steps[fwd_len:], key=lambda ns: ns[1])
        if fwd_peak_per_layer >= bwd_peak_per_layer:
            active_transient_peak, peak_step_name, peak_step_phase = fwd_peak_per_layer, fwd_peak_name, "forward"
        else:
            active_transient_peak, peak_step_name, peak_step_phase = bwd_peak_per_layer, bwd_peak_name, "backward"

        active_microbatch_peak = (layers_per_stage - 1) * persistent_per_layer + active_transient_peak
        other_in_flight_contribution = (in_flight - 1) * layers_per_stage * persistent_per_layer
        total_peak = other_in_flight_contribution + active_microbatch_peak

        optimizer_bytes = self.optimizer_state_bytes_per_rank(dp_degree, layers_per_stage)

        return {
            "persistent_per_layer": persistent_per_layer,
            "fwd_peak_per_layer": fwd_peak_per_layer,
            "bwd_peak_per_layer": bwd_peak_per_layer,
            "active_microbatch_peak": active_microbatch_peak,
            "other_in_flight_contribution": other_in_flight_contribution,
            "total_peak": total_peak,
            "optimizer_state_bytes_per_rank": optimizer_bytes,
            "total_peak_bytes": total_peak * bytes_per_element + optimizer_bytes,
            "peak_step_name": peak_step_name,
            "peak_step_phase": peak_step_phase,
        }