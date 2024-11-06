import matplotlib.pyplot as plt
import os

# Specify the path to your loss log file
loss_log_path = "/home/mxf/96_public/mxf/workgroup/huawei-chanllenge/IPcomposer/logs/lvis_337/ipcomposer-localize-lvis-1_5-1e-5/11-05-2024_14-59-15_loss_log.txt"
# loss_log_path = "/home/mxf/96_public/mxf/workgroup/huawei-chanllenge/IPcomposer/logs/ffhq/ipcomposer-localize-lvis-1_5-1e-5/11-05-2024_16-10-47_loss_log.txt"
output_dir = os.path.dirname(loss_log_path)

# Initialize lists to store steps and losses
steps = []
train_loss = []
denoise_loss = []
localization_loss = []

# Read the loss log file
with open(loss_log_path, "r") as f:
    lines = f.readlines()[1:]  # Skip the header line
    for line in lines:
        # Parse each line
        step, t_loss, d_loss, l_loss = line.strip().split()
        steps.append(int(step))
        train_loss.append(float(t_loss))
        denoise_loss.append(float(d_loss))
        localization_loss.append(float(l_loss))

# Plot and save each loss separately
# Train Loss Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, train_loss, label="Train Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Train Loss Over Steps")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_loss.png"))

# Denoise Loss Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, denoise_loss, label="Denoise Loss", color="orange")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Denoise Loss Over Steps")
plt.legend()
plt.savefig(os.path.join(output_dir, "denoise_loss.png"))

# Localization Loss Plot
plt.figure(figsize=(12, 6))
plt.plot(steps, localization_loss, label="Localization Loss", color="green")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Localization Loss Over Steps")
plt.legend()
plt.savefig(os.path.join(output_dir, "localization_loss.png"))

print(f"Plots saved to {output_dir}")
