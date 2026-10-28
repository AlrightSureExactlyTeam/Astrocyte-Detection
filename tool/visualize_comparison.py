from pathlib import Path
from real_pipeline_cli import parser_for
from real_pipeline import evaluate_experiment, generate_visualizations

if __name__ == "__main__":
    parser = parser_for("visualize")
    args = parser.parse_args()
    if args.experiment_dir is None:
        parser.error("--experiment-dir is required")
    rows = evaluate_experiment(args.data_root, args.experiment_dir, max_distance=args.match_distance)
    generate_visualizations(args.data_root, args.experiment_dir, rows)
    print(f"Figures: {Path(args.experiment_dir) / 'figures'}")
