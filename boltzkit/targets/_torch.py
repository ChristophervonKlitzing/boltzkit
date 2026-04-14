from abc import ABC, abstractmethod

from torch import nn
import torch


class TorchEval(nn.Module, ABC):

    def _get_device(self) -> torch.device | None:
        # Try parameters
        for p in self.parameters():
            return p.device
        # Fallback to buffers
        for b in self.buffers():
            return b.device
        # No params or buffers → no device
        return None

    @abstractmethod
    def _get_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _move_to_device_like(self, x: torch.Tensor):
        module_device = self._get_device()
        if module_device is not None and module_device != x.device:
            self.to(x.device)

    def get_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        self._move_to_device_like(x)
        return self._get_log_prob(x)

    def get_score(self, x: torch.Tensor) -> torch.Tensor:
        self._move_to_device_like(x)

        # Ensure x requires gradients
        x = x.clone().detach().requires_grad_(True)

        log_prob = self.get_log_prob(x)  # (batch,)

        # Compute gradient of sum of log probs w.r.t. x
        grad = torch.autograd.grad(
            outputs=log_prob.sum(), inputs=x, create_graph=False
        )[0]

        return grad  # (batch, dim)

    def get_log_prob_and_score(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._move_to_device_like(x)

        # Enable gradients on input
        x = x.clone().detach().requires_grad_(True)

        log_prob = self.get_log_prob(x)  # (batch,)

        grad = torch.autograd.grad(
            outputs=log_prob.sum(), inputs=x, create_graph=False
        )[0]

        return log_prob, grad
