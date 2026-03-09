import simtk.unit as unit
from openmm.unit import nanoseconds
from reform.simu_utils import OMMTReplicas, ReplicaExchangeHook, SimulationHook


class ObservablesHook(SimulationHook):
    def __init__(self, csv_path: str):
        super().__init__()

        self.csv_path = csv_path
        self._header_written = False
        self._re_hook = None

    def set_re_hook(self, re_hook: ReplicaExchangeHook) -> None:
        self._re_hook = re_hook

    def action(self, context: OMMTReplicas) -> None:
        states = context.get_states(getEnergy=True)

        temps = []
        pot_energies = []

        for state in states:
            e_k = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            temp = e_k * 2 / context._n_DoF / context._k
            temps.append(temp)

            pot_energies.append(
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            )

        current_step = states[0].getStepCount()
        elapsed_simulation_time = states[0].getTime().in_units_of(nanoseconds)._value

        if not self._header_written:
            with open(self.csv_path, "w") as f:
                f.write(
                    "Step,Elapsed time (ns),Exchange rate,"
                    + ",".join(f"T{i} (K)" for i in range(len(temps)))
                    + ","
                    + ",".join(f"U{i} (kJ/mol)" for i in range(len(pot_energies)))
                    + "\n"
                )
            self._header_written = True

        if self._re_hook is not None:
            exchange_rate = self._re_hook._ex_engine.exchange_rate
        else:
            exchange_rate = 0.0

        with open(self.csv_path, "a") as f:
            f.write(
                f"{current_step},{elapsed_simulation_time},{exchange_rate:.3f},"
                + ",".join(f"{t:.3f}" for t in temps)
                + ","
                + ",".join(f"{u:.3f}" for u in pot_energies)
                + "\n"
            )

    def __str__(self) -> str:
        return "Hook for observables reporting."
