import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the CSV file

experiment_name = "segmentEfficientNetVeryShortHop"

df = pd.read_csv(
    os.path.join("cks", "logs", "combined", experiment_name, "metrics.csv")
)

# Create output directory
output_dir = os.path.join("cks", "logs", "combined", experiment_name, "plots")

if not output_dir:
    output_dir = "."
elif not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set up the figure style
plt.style.use("default")
fig_size = (10, 6)

# Plot 1: Loss plot
plt.figure(figsize=fig_size)
plt.plot(
    df["epoch"],
    df["train0_loss"],
    label="Train0 Loss",
    marker="o",
    markersize=4,
    alpha=0.7,
)
plt.plot(
    df["epoch"],
    df["train1_loss"],
    label="Train1 Loss",
    marker="s",
    markersize=4,
    alpha=0.7,
)
plt.plot(
    df["epoch"],
    df["valid_loss"],
    label="Validation Loss",
    marker="^",
    markersize=4,
    alpha=0.7,
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: Accuracy plot
plt.figure(figsize=fig_size)
plt.plot(
    df["epoch"],
    df["train0_acc"],
    label="Train0 Accuracy",
    marker="o",
    markersize=4,
    alpha=0.7,
)
plt.plot(
    df["epoch"],
    df["train1_acc"],
    label="Train1 Accuracy",
    marker="s",
    markersize=4,
    alpha=0.7,
)
plt.plot(
    df["epoch"],
    df["valid_acc"],
    label="Validation Accuracy",
    marker="^",
    markersize=4,
    alpha=0.7,
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"Plots saved successfully in: {os.path.abspath(output_dir)}")
print("- loss_plot.png")
print("- accuracy_plot.png")
