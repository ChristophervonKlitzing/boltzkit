import time

from openmm.unit import nanoseconds
from reform.simu_utils import OMMTReplicas, SimulationHook


class StatusHook(SimulationHook):
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        super().__init__()

        self.start_time = None

    def start_timer(self):
        self.start_time = time.time()

    def action(self, context: OMMTReplicas) -> None:
        current_step = context.get_state(0).getStepCount()

        report_message = f"Step {current_step}/{self.total_steps} ({current_step/self.total_steps*100:.2f}%)"

        if self.start_time is not None:
            elapsed_time = (time.time() - self.start_time) / (3600 * 24)  # in days
            elapsed_simulation_time = (
                context.get_state(0).getTime().in_units_of(nanoseconds)._value
            )
            report_message += (
                f" | Speed (ns/day): {elapsed_simulation_time / elapsed_time:.2f}"
            )

        print(report_message, flush=True)

    def __str__(self) -> str:
        return "Hook for status reporting."
