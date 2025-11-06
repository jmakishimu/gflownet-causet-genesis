# Add these alternative beta schedule functions to train.py
# Replace the get_adaptive_beta function with one of these

import numpy as np

# OPTION 1: Linear schedule (simplest, most stable)
def get_adaptive_beta_linear(step: int, max_steps: int,
                             start: float = 0.1, end: float = 1.0) -> float:
    """
    Linear beta schedule: β increases at constant rate
    Most stable for learning
    """
    progress = min(step / max_steps, 1.0)
    return start + (end - start) * progress


# OPTION 2: Square root (slower than exponential, faster than linear)
def get_adaptive_beta_sqrt(step: int, max_steps: int,
                           start: float = 0.1, end: float = 1.0) -> float:
    """
    Square root schedule: β increases quickly at first, then slows
    Good balance between exploration and exploitation
    """
    progress = min(step / max_steps, 1.0)
    return start + (end - start) * np.sqrt(progress)


# OPTION 3: Logarithmic (very gentle, stays low for long time)
def get_adaptive_beta_log(step: int, max_steps: int,
                          start: float = 0.1, end: float = 1.0) -> float:
    """
    Logarithmic schedule: β stays low for most of training
    Maximum exploration time
    """
    progress = min(step / max_steps, 1.0)
    if progress < 0.01:
        return start
    # Use log scale from 0.01 to 1.0
    log_progress = np.log(1 + 99 * progress) / np.log(100)
    return start + (end - start) * log_progress


# OPTION 4: Warmup + Linear (recommended!)
def get_adaptive_beta_warmup_linear(step: int, max_steps: int,
                                   start: float = 0.1, end: float = 1.0,
                                   warmup_frac: float = 0.3) -> float:
    """
    Warmup + linear schedule:
    - First 30%: β stays constant (pure exploration)
    - Last 70%: β increases linearly (gradual exploitation)

    This is BEST for GFlowNets - gives model time to learn structure
    before applying strong penalties
    """
    warmup_steps = int(max_steps * warmup_frac)

    if step < warmup_steps:
        return start
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return start + (end - start) * progress


# OPTION 5: Current (exponential) - for comparison
def get_adaptive_beta_exponential(step: int, max_steps: int,
                                 start: float = 0.1, end: float = 1.0) -> float:
    """
    Exponential schedule (current approach)
    Problem: β grows too fast at the end
    """
    progress = min(step / max_steps, 1.0)
    return start * (end / start) ** progress


# Visualization code to compare schedules
def visualize_schedules():
    """Run this to see how different schedules behave"""
    import matplotlib.pyplot as plt

    steps = np.arange(0, 1000)

    schedules = {
        'Linear': [get_adaptive_beta_linear(s, 1000, 0.1, 1.0) for s in steps],
        'Square Root': [get_adaptive_beta_sqrt(s, 1000, 0.1, 1.0) for s in steps],
        'Logarithmic': [get_adaptive_beta_log(s, 1000, 0.1, 1.0) for s in steps],
        'Warmup+Linear (30%)': [get_adaptive_beta_warmup_linear(s, 1000, 0.1, 1.0, 0.3) for s in steps],
        'Exponential (current)': [get_adaptive_beta_exponential(s, 1000, 0.1, 1.0) for s in steps],
    }

    plt.figure(figsize=(12, 6))
    for name, values in schedules.items():
        plt.plot(steps, values, label=name, linewidth=2)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Beta Value', fontsize=12)
    plt.title('Comparison of Beta Schedules', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('beta_schedules_comparison.png', dpi=150)
    print("Saved comparison to beta_schedules_comparison.png")

    # Print key statistics
    print("\nBeta values at key points (start=0.1, end=1.0, steps=1000):")
    print(f"{'Schedule':<25} | Step 0 | Step 250 | Step 500 | Step 750 | Step 1000")
    print("-" * 80)
    for name, values in schedules.items():
        print(f"{name:<25} | {values[0]:6.3f} | {values[250]:8.3f} | {values[500]:8.3f} | {values[750]:8.3f} | {values[999]:9.3f}")


if __name__ == "__main__":
    visualize_schedules()
