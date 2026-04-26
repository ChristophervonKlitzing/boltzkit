import unittest
import torch
import jax
import jax.numpy as jnp
import numpy as np
from typing import TYPE_CHECKING

from boltzkit.utils.framework import make_agnostic

if TYPE_CHECKING:
    from boltzkit.utils.framework import UnionArrayType


def log_prob_np(x: np.ndarray):  # batched
    return -np.sum(x**2, -1)


def score_np(x: np.ndarray):  # batched
    return -2 * x


def log_prob_jax(x: jax.Array):  # scalar function
    return -jnp.sum(x**2)


def log_prob_torch(x: torch.Tensor):  # batched
    return -torch.sum(x**2, -1)


def _to_numpy(x):
    if isinstance(x, (torch.Tensor, jax.Array)):
        x = np.asarray(x)
    return x


class TestAgnosticFunctions(unittest.TestCase):
    def setUp(self):
        # use dtype that is not default in jax, pytorch, or numpy to test dtype consistency
        self.x_np = np.asarray([[1, 2], [3, -0.5], [-10, 9.5]], dtype=np.float16)
        self.x_torch = torch.from_numpy(self.x_np)
        self.x_jax = jnp.asarray(self.x_np)

        # Map implementations to their functions
        self.funcs = {
            "numpy": make_agnostic(
                implementation="numpy", value_fn=log_prob_np, grad_fn=score_np
            ),
            "torch": make_agnostic(implementation="pytorch", value_fn=log_prob_torch),
            "jax": make_agnostic(implementation="jax", value_fn=log_prob_jax),
        }

        # Map array types
        self.arrays: dict[str, "UnionArrayType"] = {
            "numpy": self.x_np,
            "torch": self.x_torch,
            "jax": self.x_jax,
        }

    def test_all_combinations_log_prob(self):
        # Compare against the reference: numpy function on numpy array
        ref = log_prob_np(self.x_np)

        # Loop over all combinations: array type x function implementation
        for arr_name, arr in self.arrays.items():
            for func_name, func in self.funcs.items():
                # Convert result to NumPy for comparison
                result = func(arr)
                self.assertEqual(type(result), type(arr))
                self.assertEqual(result.dtype, arr.dtype)

                if isinstance(arr, torch.Tensor):
                    self.assertEqual(result.device, arr.device)

                result = _to_numpy(result)

                np.testing.assert_allclose(
                    result,
                    ref,
                    err_msg=f"Failed for array={arr_name}, func={func_name}",
                )

    def test_all_combinations_score(self):
        # Compare against the reference: numpy function on numpy array
        ref = score_np(self.x_np)

        # Loop over all combinations: array type x function implementation
        for arr_name, arr in self.arrays.items():
            for func_name, func in self.funcs.items():
                # Convert result to NumPy for comparison
                result = func.get_grad(arr)
                self.assertEqual(type(result), type(arr))
                self.assertEqual(result.dtype, arr.dtype)

                if isinstance(arr, torch.Tensor):
                    self.assertEqual(result.device, arr.device)

                result = _to_numpy(result)

                np.testing.assert_allclose(
                    result,
                    ref,
                    err_msg=f"Failed for array={arr_name}, func={func_name}",
                )

    def test_all_combinations_score(self):
        # Compare against the reference: numpy function on numpy array
        ref_log_prob = log_prob_np(self.x_np)
        ref_score = score_np(self.x_np)

        # Loop over all combinations: array type x function implementation
        for arr_name, arr in self.arrays.items():
            for func_name, func in self.funcs.items():
                # Convert result to NumPy for comparison
                result_log_prob, result_score = func.get_value_and_grad(arr)
                self.assertEqual(type(result_log_prob), type(arr))
                self.assertEqual(type(result_score), type(arr))
                self.assertEqual(result_log_prob.dtype, arr.dtype)
                self.assertEqual(result_score.dtype, arr.dtype)

                if isinstance(arr, torch.Tensor):
                    self.assertEqual(result_log_prob.device, arr.device)
                    self.assertEqual(result_score.device, arr.device)

                result_log_prob = _to_numpy(result_log_prob)
                result_score = _to_numpy(result_score)

                np.testing.assert_allclose(
                    result_log_prob,
                    ref_log_prob,
                    err_msg=f"Failed for array={arr_name}, func={func_name}",
                )
                np.testing.assert_allclose(
                    result_score,
                    ref_score,
                    err_msg=f"Failed for array={arr_name}, func={func_name}",
                )

    def test_torch_autograd1(self):
        x_torch = self.x_torch.clone()
        x_torch.requires_grad = True

        def trafo(x: torch.Tensor):
            return 2 * x + x**2 + 1

        y = trafo(x_torch)

        ref_score_y = score_np(y.detach().numpy())

        # Test score computation if y is no leaf variable
        for func_name, func in self.funcs.items():
            result_score_y = func.get_grad(y).numpy()  # should already be detached
            np.testing.assert_allclose(
                result_score_y,
                ref_score_y,
                err_msg=f"Failed for func={func_name}",
            )

    def test_torch_autograd2(self):
        """
        This test checks if autograd works through the wrapped transformation with transformations
        before (trafo) and after (.sum()), if the input is a torch.Tensor with additional score-computation.
        """

        def reset_requirements():
            x_torch = self.x_torch.clone()
            x_torch.requires_grad = True

            def trafo(x: torch.Tensor):
                return 2 * x + x**2 + 1

            y = trafo(x_torch)
            return x_torch, y

        # Compute grad for x_torch using the pytorch implementation
        x_torch, y = reset_requirements()
        log_prob_torch(y).sum().backward()
        ref_grad_x = x_torch.grad.detach().numpy()
        ref_score = score_np(y.detach().numpy())

        # Test score computation if y is no leaf variable
        for func_name, func in self.funcs.items():
            # overwrites the variables, allowing multiple backward passes
            x_torch, y = reset_requirements()

            # Test in the presence of score computation if autograd still works
            result_value, result_score = func.get_value_and_grad(y)
            result_value.sum().backward()
            result_grad_x = x_torch.grad.detach().numpy()

            np.testing.assert_allclose(
                result_grad_x,
                ref_grad_x,
                err_msg=f"Failed for func={func_name}",
            )

            # To make sure the score computation isn't affected
            np.testing.assert_allclose(
                result_score.numpy(),
                ref_score,
                err_msg=f"Failed for func={func_name}",
            )


if __name__ == "__main__":
    unittest.main()
