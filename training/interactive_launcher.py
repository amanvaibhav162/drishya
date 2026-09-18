#!/usr/bin/env python3
"""
DRISHYA SOTA Training: Interactive Configuration Wizard & CLI Launcher
Provides an intuitive, guided terminal interface for configuring and launching
the 7-stage training pipeline on the 24 GB Quadro RTX 6000 foundry.
"""

import os
import sys
import subprocess
import shutil

# ANSI Color Palette
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def clear_screen():
    print("\033[H\033[J", end="")


def print_banner():
    print(f"{CYAN}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════════════════╗")
    print("  ║         दृष्य  DRISHYA SOTA Training Configuration Wizard        ║")
    print("  ║         Hardware Target: NVIDIA Quadro RTX 6000 (24 GB)          ║")
    print("  ║          Autonomous Resilient Multi-Stage Deep Learning          ║")
    print("  ╚══════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")


def get_gpu_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "total_mem": parts[1],
            "free_mem": parts[2],
            "driver": parts[3]
        }
    except Exception:
        return None


def prompt_int(prompt_text, default_val, min_val=1, max_val=1000):
    while True:
        user_input = input(f"{BOLD}{prompt_text}{RESET} [{CYAN}{default_val}{RESET}]: ").strip()
        if not user_input:
            return default_val
        try:
            val = int(user_input)
            if min_val <= val <= max_val:
                return val
            print(f"{YELLOW}⚠ Value must be between {min_val} and {max_val}.{RESET}")
        except ValueError:
            print(f"{RED}✖ Please enter a valid integer.{RESET}")


def prompt_choice(prompt_text, options, default_idx=0):
    print(f"\n{BOLD}{prompt_text}:{RESET}")
    for i, opt in enumerate(options):
        marker = f"{CYAN}*{RESET}" if i == default_idx else " "
        print(f"  [{CYAN}{i+1}{RESET}] {opt['label']} {DIM}({opt['desc']}){RESET}")
    
    while True:
        user_input = input(f"\n{BOLD}Select option (1-{len(options)}){RESET} [{CYAN}{default_idx+1}{RESET}]: ").strip()
        if not user_input:
            return options[default_idx]["value"]
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(options):
                return options[idx]["value"]
            print(f"{YELLOW}⚠ Choice must be between 1 and {len(options)}.{RESET}")
        except ValueError:
            print(f"{RED}✖ Invalid selection.{RESET}")


def prompt_yes_no(prompt_text, default_yes=True):
    default_str = "Y/n" if default_yes else "y/N"
    choice = input(f"{BOLD}{prompt_text}{RESET} [{CYAN}{default_str}{RESET}]: ").strip().lower()
    if not choice:
        return default_yes
    return choice in ["y", "yes"]


def main():
    clear_screen()
    print_banner()

    # 1. System Hardware Audit
    gpu = get_gpu_info()
    print(f"{BOLD}📊 Detected Hardware Environment:{RESET}")
    if gpu:
        print(f"   • GPU:        {GREEN}{gpu['name']}{RESET} ({gpu['total_mem']} VRAM, {gpu['free_mem']} Free)")
        print(f"   • Driver:     {DIM}{gpu['driver']}{RESET}")
    else:
        print(f"   • GPU:        {YELLOW}No active GPU detected via nvidia-smi (CPU Mode){RESET}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"   • Root Path:  {DIM}{project_root}{RESET}\n")

    # 2. Stage Selection
    stage_options = [
        {"value": "all", "label": "Full 7-Stage SOTA Pipeline (Recommended)", "desc": "Teachers 1-3 -> Pseudo-Labels -> Distillation -> XGBoost -> ONNX"},
        {"value": "teacher1", "label": "Stage 1: Teacher 1 (ConvNeXtV2-Base)", "desc": "Microaneurysms specialist, ~4.5 hrs"},
        {"value": "teacher2", "label": "Stage 2: Teacher 2 (Swin-Base-384)", "desc": "Quadrant relationship specialist, ~4.8 hrs"},
        {"value": "teacher3", "label": "Stage 3: Teacher 3 (EfficientNet-B5)", "desc": "Exudate boundary specialist, ~3.8 hrs"},
        {"value": "pseudo", "label": "Stage 4: Consensus Pseudo-Labeling", "desc": "Generates 3-teacher soft labels into cache/pseudo_labels.h5, ~45 mins"},
        {"value": "distill", "label": "Stage 5: Student Knowledge Distillation", "desc": "Trains 22M EfficientNetV2-S U-Net, ~3.5 hrs"},
        {"value": "xgboost", "label": "Stage 6: 32-D Morphometry & XGBoost Fit", "desc": "Feature extraction + Nelder-Mead Kappa thresholds, ~25 mins"},
        {"value": "export", "label": "Stage 7: Production ONNX Export", "desc": "FP16 dynamic ONNX conversion and edge latency audit, ~5 mins"}
    ]
    stage = prompt_choice("1. Select Target Execution Stage", stage_options, default_idx=0)

    # 3. Hyperparameters
    print(f"\n{BOLD}2. Configure Training Hyperparameters:{RESET}")
    fold = prompt_int("   • Validation Holdout Fold (0-4)", default_val=0, min_val=0, max_val=4)
    epochs = prompt_int("   • Training Epochs per Stage", default_val=30, min_val=1, max_val=100)
    batch_size = prompt_int("   • Physical Batch Size (Calibrated for 24 GB VRAM)", default_val=8, min_val=1, max_val=32)
    accum_steps = prompt_int("   • Gradient Accumulation Steps", default_val=2, min_val=1, max_val=16)
    workers = prompt_int("   • DataLoader Worker Threads", default_val=8, min_val=0, max_val=16)
    effective_batch = batch_size * accum_steps

    # 4. Resilience & Execution Options
    print(f"\n{BOLD}3. Runtime Execution Mode:{RESET}")
    launch_options = [
        {"value": "tmux", "label": "Persistent tmux Session (Recommended)", "desc": "Runs safely in background; survive SSH drops & laptop sleep"},
        {"value": "foreground", "label": "Direct Foreground Console", "desc": "Live progress bar in current terminal tab"}
    ]
    launch_mode = prompt_choice("Select Execution Environment", launch_options, default_idx=0)

    force_retrain = False
    if stage != "all":
        force_retrain = prompt_yes_no("Force re-run even if target model checkpoint already exists?", default_yes=False)

    # 5. Review Configuration Summary
    clear_screen()
    print_banner()
    print(f"{BOLD}📋 Review Selected Configuration Summary:{RESET}")
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ Target Pipeline:   {CYAN}{stage.upper():<41}{RESET}│")
    print(f"  │ Holdout Lockbox:   {GREEN}Fold {fold:<36}{RESET}│")
    print(f"  │ Training Epochs:   {epochs:<41}│")
    print(f"  │ Physical Batch:    {batch_size} (Accum: {accum_steps} -> Effective: {effective_batch:<16})│")
    print(f"  │ DataLoader Workers:{workers:<41}│")
    print(f"  │ Arithmetic:        {GREEN}Native FP16 Mixed Precision (Turing TU102){RESET} │")
    print(f"  │ Execution Target:  {CYAN}{launch_mode.upper():<41}{RESET}│")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    confirm = prompt_yes_no("\n🚀 Ready to launch training with these settings?", default_yes=True)
    if not confirm:
        print(f"\n{YELLOW}Training launch cancelled by user.{RESET}\n")
        sys.exit(0)

    # Build execution command
    runner_script = os.path.join(project_root, "training", "run_sota_training.sh")
    cmd_args = [
        runner_script,
        "--stage", stage,
        "--fold", str(fold),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--accum_steps", str(accum_steps),
        "--workers", str(workers),
        "--splits_csv", "data/splits/5fold_splits.csv",
        "--data_dir", "."
    ]
    if force_retrain:
        cmd_args.append("--force")

    cmd_str = " ".join(cmd_args)

    if launch_mode == "tmux":
        session_name = "drishya_training"
        # Check if tmux is installed
        if not shutil.which("tmux"):
            print(f"{YELLOW}[WARNING] tmux is not installed. Falling back to foreground execution.{RESET}")
            subprocess.run(cmd_args, cwd=project_root)
            return

        print(f"\n{GREEN}[INFO] Spawning training inside persistent tmux session '{session_name}'...{RESET}")
        
        # Kill previous session if user wants fresh one or attach
        check_tmux = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
        if check_tmux.returncode == 0:
            replace = prompt_yes_no(f"A tmux session '{session_name}' already exists. Kill and replace it?", default_yes=True)
            if replace:
                subprocess.run(["tmux", "kill-session", "-t", session_name])
            else:
                print(f"{YELLOW}Attaching to existing session...{RESET}")
                os.system(f"tmux attach -t {session_name}")
                return

        # Start new tmux detached session running the command wrapped in bash
        shell_cmd = f"cd {project_root} && {cmd_str}; exec bash"
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, "bash", "-c", shell_cmd])

        print(f"\n{GREEN}✔ Training pipeline successfully launched inside tmux session '{session_name}'!{RESET}\n")
        print(f"{BOLD}Quick Control Commands:{RESET}")
        print(f"  • Attach to live console:  {CYAN}tmux attach -t {session_name}{RESET}")
        print(f"  • Detach safely anytime:   {DIM}Ctrl + B, then D{RESET}")
        print(f"  • Stream execution log:    {CYAN}tail -f logs/sota_training_*.log{RESET}")
        print(f"  • Monitor GPU utilization: {CYAN}watch -n 1 nvidia-smi{RESET}\n")

        attach_now = prompt_yes_no("Attach to the live tmux console now?", default_yes=True)
        if attach_now:
            os.system(f"tmux attach -t {session_name}")

    else:
        print(f"\n{GREEN}[INFO] Launching training in foreground...{RESET}\n")
        try:
            subprocess.run(cmd_args, cwd=project_root)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}[ALERT] Training interrupted by user. Checkpoints preserved.{RESET}")


if __name__ == "__main__":
    main()
