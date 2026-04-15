from typing import Callable, NamedTuple

import jax


class JaxEval(NamedTuple):
    get_log_prob: Callable[[jax.Array], jax.Array]
    get_score: Callable[[jax.Array], jax.Array]
    get_log_prob_and_score: Callable[[jax.Array], tuple[jax.Array, jax.Array]]


def make_eval_from_jax_log_prob_single(
    jax_log_prob_single: Callable[[jax.Array], jax.Array],
) -> JaxEval:
    # use jit, vmap and grad
    get_log_prob = jax.vmap(jax_log_prob_single)
    get_score = jax.vmap(jax.grad(jax_log_prob_single))
    get_log_prob_and_score = jax.vmap(jax.value_and_grad(jax_log_prob_single))

    return JaxEval(
        get_log_prob=jax.jit(get_log_prob),
        get_score=jax.jit(get_score),
        get_log_prob_and_score=jax.jit(get_log_prob_and_score),
    )
