from boltzkit.targets.base.dispatched_eval.jax import JaxEval
import jax


def create_custom_jax_eval():  # optional arguments
    def jax_log_prob_single(x: jax.Array):  # x is NOT batched in this case
        return -(x**2)

    return JaxEval.create_from_log_prob_single(jax_log_prob_single)
