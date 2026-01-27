#!/usr/bin/env python3
"""
Generate bar chart figures for CausalFlow paper.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a clean style
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.linewidth'] = 0.8

# Data from the paper
benchmarks = ['GSM8K', 'MBPP', 'SealQA\nHard', 'MedBrowse\nComp']
benchmarks_short = ['GSM8K', 'MBPP', 'SealQA', 'MedBrowse']

# Repair rates (%)
repair_rates = [52.4, 41.2, 21.9, 44.5]

# Accuracy (%)
baseline_accuracy = [75.0, 55.2, 42.5, 30.8]
post_repair_accuracy = [88.1, 76.4, 55.1, 61.6]

# Minimality scores
minimality_scores = [0.87, 0.82, 0.79, 0.84]

# Colors matching the image
teal_color = '#2E8B8B'  # Teal/dark cyan
orange_color = '#E07020'  # Orange

def create_repair_rate_chart():
    """Create repair rate bar chart."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    x = np.arange(len(benchmarks))
    bars = ax.bar(x, repair_rates, color=teal_color, width=0.6, edgecolor='none')

    ax.set_ylabel('Repair Rate (%)', fontweight='bold')
    ax.set_title('Repair Rate (↑)', fontweight='bold', loc='left', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_short, fontsize=9)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])

    # Add value labels on bars
    for bar, val in zip(bars, repair_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/repair_rate.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/repair_rate.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created repair_rate.pdf")

def create_accuracy_chart():
    """Create baseline vs post-repair accuracy bar chart."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    x = np.arange(len(benchmarks))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_accuracy, width, label='Baseline',
                   color=orange_color, edgecolor='none')
    bars2 = ax.bar(x + width/2, post_repair_accuracy, width, label='Post-repair',
                   color=teal_color, edgecolor='none')

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy (baseline → post-repair)', fontweight='bold', loc='left', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_short, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/accuracy_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/accuracy_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created accuracy_comparison.pdf")

def create_combined_figure():
    """Create a combined figure with both charts side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    x = np.arange(len(benchmarks_short))

    # Left: Repair Rate
    bars1 = ax1.bar(x, repair_rates, color=teal_color, width=0.6, edgecolor='none')
    ax1.set_title('Repair Rate (↑)', fontweight='bold', loc='left', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks_short, fontsize=9)
    ax1.set_ylim(0, 60)
    ax1.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax1.set_ylabel('Repair Rate (%)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)

    # Add value labels
    for bar, val in zip(bars1, repair_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    # Right: Accuracy Comparison
    width = 0.35
    bars2 = ax2.bar(x - width/2, baseline_accuracy, width, label='Baseline',
                    color=orange_color, edgecolor='none')
    bars3 = ax2.bar(x + width/2, post_repair_accuracy, width, label='Post-repair',
                    color=teal_color, edgecolor='none')

    ax2.set_title('Accuracy (baseline → post-repair)', fontweight='bold', loc='left', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks_short, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='upper right', frameon=False, fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/results_combined.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/results_combined.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created results_combined.pdf")

def create_minimality_chart():
    """Create minimality scores bar chart."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    x = np.arange(len(benchmarks_short))
    bars = ax.bar(x, minimality_scores, color=teal_color, width=0.6, edgecolor='none')

    ax.set_ylabel('Minimality Score')
    ax.set_title('Minimality (↑)', fontweight='bold', loc='left', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks_short, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Add value labels on bars
    for bar, val in zip(bars, minimality_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('figures/minimality.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/minimality.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created minimality.pdf")

def create_three_panel_figure():
    """Create a three-panel figure with repair rate, accuracy, and minimality."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

    x = np.arange(len(benchmarks_short))

    # Panel 1: Repair Rate
    bars1 = ax1.bar(x, repair_rates, color=teal_color, width=0.6, edgecolor='none')
    ax1.set_title('(a) Repair Rate (↑)', fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks_short, fontsize=9)
    ax1.set_ylim(0, 60)
    ax1.set_ylabel('Rate (%)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax1.set_axisbelow(True)
    for bar, val in zip(bars1, repair_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    # Panel 2: Accuracy Comparison
    width = 0.35
    bars2 = ax2.bar(x - width/2, baseline_accuracy, width, label='Baseline',
                    color=orange_color, edgecolor='none')
    bars3 = ax2.bar(x + width/2, post_repair_accuracy, width, label='Post-repair',
                    color=teal_color, edgecolor='none')
    ax2.set_title('(b) Accuracy (↑)', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(benchmarks_short, fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax2.set_axisbelow(True)

    # Panel 3: Minimality
    bars4 = ax3.bar(x, minimality_scores, color=teal_color, width=0.6, edgecolor='none')
    ax3.set_title('(c) Minimality (↑)', fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(benchmarks_short, fontsize=9)
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel('Score')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, linestyle='-', alpha=0.3)
    ax3.set_axisbelow(True)
    for bar, val in zip(bars4, minimality_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig('figures/results_three_panel.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/results_three_panel.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Created results_three_panel.pdf")

if __name__ == '__main__':
    import os

    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)

    # Generate all figures
    create_repair_rate_chart()
    create_accuracy_chart()
    create_combined_figure()
    create_minimality_chart()
    create_three_panel_figure()

    print("\nAll figures generated in figures/ directory")
