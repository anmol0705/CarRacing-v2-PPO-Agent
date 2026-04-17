"""Export training metrics from log file to CSV for the Streamlit dashboard."""

import csv
import re
import sys
from pathlib import Path


def parse_log(log_path: str) -> tuple[list[dict], list[dict]]:
    """Parse training log into train metrics and eval metrics."""
    train_rows = []
    eval_rows = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # Parse training lines:
            # Step  4861952 | Reward:   114.9 | Policy: -0.0030 | Value: 6.3828 | Ent: 0.9798 | KL: 0.0011 | Clip: 0.007 | LR: 0.000025 | SPS: 144
            m = re.match(
                r"Step\s+(\d+)\s+\|\s+Reward:\s+([\d\.\-]+)\s+\|\s+"
                r"Policy:\s+([\d\.\-]+)\s+\|\s+Value:\s+([\d\.\-]+)\s+\|\s+"
                r"Ent:\s+([\d\.\-]+)\s+\|\s+KL:\s+([\d\.\-]+)\s+\|\s+"
                r"Clip:\s+([\d\.\-]+)\s+\|\s+LR:\s+([\d\.\-]+)\s+\|\s+"
                r"SPS:\s+([\d\.\-]+)",
                line,
            )
            if m:
                train_rows.append({
                    "step": int(m.group(1)),
                    "ep_reward_mean": float(m.group(2)),
                    "policy_loss": float(m.group(3)),
                    "value_loss": float(m.group(4)),
                    "entropy": float(m.group(5)),
                    "approx_kl": float(m.group(6)),
                    "clip_frac": float(m.group(7)),
                    "lr": float(m.group(8)),
                    "sps": float(m.group(9)),
                })

            # Parse eval lines:
            # EVAL @ 4950016: 806.0 +/- 233.5
            m_eval = re.match(r"\s*EVAL @ (\d+):\s+([\d\.\-]+)\s+\+/-\s+([\d\.\-]+)", line)
            if m_eval:
                eval_rows.append({
                    "step": int(m_eval.group(1)),
                    "eval_reward": float(m_eval.group(2)),
                    "eval_std": float(m_eval.group(3)),
                })

    return train_rows, eval_rows


def main():
    # Find the training log
    log_candidates = [
        "logs/train_v5.log",
        "logs/train.log",
    ]
    log_path = None
    for candidate in log_candidates:
        if Path(candidate).exists():
            log_path = candidate
            break

    if not log_path:
        print("ERROR: No training log found")
        sys.exit(1)

    print(f"Parsing {log_path}...")
    train_rows, eval_rows = parse_log(log_path)
    print(f"  Found {len(train_rows)} training entries, {len(eval_rows)} eval entries")

    # Write training metrics CSV
    Path("assets").mkdir(exist_ok=True)

    with open("assets/training_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step", "ep_reward_mean", "policy_loss", "value_loss",
            "entropy", "approx_kl", "clip_frac", "lr", "sps",
        ])
        writer.writeheader()
        writer.writerows(train_rows)

    with open("assets/eval_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "eval_reward", "eval_std"])
        writer.writeheader()
        writer.writerows(eval_rows)

    print(f"  Saved assets/training_metrics.csv ({len(train_rows)} rows)")
    print(f"  Saved assets/eval_metrics.csv ({len(eval_rows)} rows)")


if __name__ == "__main__":
    main()
