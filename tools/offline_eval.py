import argparse
from pathlib import Path

from boltzkit.evaluation.molecular_eval import TicaEval, TorsionMarginalEval
from boltzkit.utils.dataloader import load_from_file, load_tica_model, load_topology
from boltzkit.evaluation.eval import (
    EvalData,
    NllEval,
    EnergyHistEval,
    get_scalar_metrics,
    run_eval,
    get_pdfs,
    get_histograms,
)
from boltzkit.utils.histogram import save_histograms
from boltzkit.utils.pdf import plot_pdf, save_pdfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate offline by loading samples and log probs from files."
    )

    parser.add_argument(
        "--true_samples",
        type=Path,
        default=None,
        help="Path to the true (e.g. MD) samples",
    )

    parser.add_argument(
        "--true_samples_target_log_prob",
        type=Path,
        default=None,
        help="Path to the target log probs of the true samples",
    )

    parser.add_argument(
        "--pred_samples",
        default=None,
        type=Path,
        help="Path to the predicted samples",
    )

    parser.add_argument(
        "--pred_samples_target_log_prob",
        default=None,
        type=Path,
        help="Path to the target log probs of the predicted samples",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        help="Specify the number of samples and log probs to use",
    )

    choices = ["energy_hist", "tica", "torsion_marginals"]
    default = choices[:1].copy()
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        choices=choices,
        default=default,
        help="One or more evaluation metrics to run. Duplicates will be ignored.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="The output directory, where files are stored to",
    )

    parser.add_argument(
        "--tica_model",
        type=Path,
        default=None,
        help="Path to a tica model (e.g., a pickled deeptime model)",
    )

    parser.add_argument(
        "--topology",
        type=Path,
        default=None,
        help="Path to a topology file (e.g., a pdb file)",
    )

    args = parser.parse_args()

    # Check optional dependencies
    if "tica" in args.eval:
        if args.tica_model is None:
            parser.error(
                "Argument --tica_model is required if 'tica' is included in --eval"
            )
        if args.topology is None:
            parser.error(
                "Argument --topology is required if 'tica' is included in --eval"
            )

    if "torsion_marginals" in args.eval:
        if args.topology is None:
            parser.error(
                "Argument --topology is required if 'tica' is included in --eval"
            )

    return args


def main():
    args = parse_args()

    if args.true_samples is not None:
        true_samples = load_from_file(
            args.true_samples, data_type="samples", n_samples=args.n_samples
        )
    else:
        true_samples = None

    if args.true_samples_target_log_prob is not None:
        true_samples_target_log_prob = load_from_file(
            args.true_samples_target_log_prob,
            data_type="log_probs",
            n_samples=args.n_samples,
        )
    else:
        true_samples_target_log_prob = None

    if args.pred_samples is not None:
        pred_samples = load_from_file(
            args.pred_samples, data_type="samples", n_samples=args.n_samples
        )
    else:
        pred_samples = None

    if args.pred_samples_target_log_prob is not None:
        pred_samples_target_log_prob = load_from_file(
            args.pred_samples_target_log_prob,
            data_type="log_probs",
            n_samples=args.n_samples,
        )
    else:
        pred_samples_target_log_prob = None

    outdir: Path | None = args.outdir
    if outdir is not None and not outdir.exists():
        raise FileNotFoundError(
            f"The specified output directory does not exist: '{outdir}'. "
            "Please create it or provide a valid path."
        )

    eval_modes = list(set(args.eval))

    # Create evaluation pipeline from the specified eval nodes
    eval_objects = []
    for mode in eval_modes:
        if mode == "nll":
            eval_objects.append(NllEval())
        elif mode == "energy_hist":
            eval_objects.append(EnergyHistEval())
        elif mode == "tica":
            topology = load_topology(args.topology)
            tica_model = load_tica_model(args.tica_model)
            eval_objects.append(TicaEval(topology=topology, tica_model=tica_model))
        elif mode == "torsion_marginals":
            topology = load_topology(args.topology)
            eval_objects.append(TorsionMarginalEval(topology=topology))
        else:
            raise ValueError(f"Unknown eval mode '{mode}'")

    # Run the evaluation pipeline
    data = EvalData(
        samples_true=true_samples,
        samples_pred=pred_samples,
        true_samples_target_log_prob=true_samples_target_log_prob,
        pred_samples_target_log_prob=pred_samples_target_log_prob,
    )
    results = run_eval(data, evals=eval_objects, skip_on_missing_data=False)

    scalar_metrics = get_scalar_metrics(results)
    pdfs = get_pdfs(results)

    # Optionally save output to directory
    if outdir is not None:
        save_pdfs(pdfs, outdir.as_posix())

        hists = get_histograms(results)
        save_histograms(hists, outdir.as_posix())

        # TODO: Store metrics in a yaml

    if len(pdfs) > 0:
        for key, pdf_buffer in pdfs.items():
            plot_pdf(pdf_buffer, dpi=100, show=True, title=key.replace("_pdf", ""))

    # Print scalar metrics if any
    if len(scalar_metrics) > 0:
        print("\n=== Evaluation Metrics ===\n")

        for k, v in scalar_metrics.items():
            if isinstance(v, (float, int)):
                print(f"{k:30s}: {v:.6f}")
            else:
                print(f"{k:30s}: {v}")


if __name__ == "__main__":
    main()
