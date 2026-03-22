from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_scaling_curves(summary_csv_path: str, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv_path)
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task]
        plt.figure(figsize=(7, 4.5))
        for fmt in ['answer_only', 'short_rationale', 'full_rationale']:
            sub = task_df[task_df['supervision_format'] == fmt].sort_values('budget')
            if sub.empty:
                continue
            plt.plot(sub['budget'], sub['exact_match_mean'], marker='o', label=fmt)
            plt.fill_between(
                sub['budget'],
                sub['exact_match_mean'] - sub['exact_match_std'],
                sub['exact_match_mean'] + sub['exact_match_std'],
                alpha=0.15,
            )
        plt.title(f"{task}: Exact Match vs Tiny Budget")
        plt.xlabel('Budget (examples)')
        plt.ylabel('Exact match')
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f'scaling_{task}.png', dpi=140)
        plt.close()
