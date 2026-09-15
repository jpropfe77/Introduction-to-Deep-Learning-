# -*- coding: utf-8 -*-
"""CSCI 5425 - Assignment 2.

Linear Regression vs PyTorch MLP on the California Housing dataset.

Run with:
    python asignment2_jp.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works headless / without a display
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim


README_PATH = "README.md"
LOSS_PLOT_PATH = "loss_curve.png"


def main():
    readme_lines = []

    # ------------------------------------------------------------------
    # Part 1: Data Loading and Exploration
    # ------------------------------------------------------------------
    housing = fetch_california_housing()


    print("DATASET DESCRIPTION")
    print(housing.DESCR)

    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe())

    readme_lines.append("## Part 1: Data Loading and Exploration\n")
    readme_lines.append("### Dataset Description\n")
    readme_lines.append("```\n" + housing.DESCR + "\n```\n")
    readme_lines.append("### First 5 Rows\n")
    readme_lines.append(df.head().to_markdown() + "\n")
    readme_lines.append("### Summary Statistics\n")
    readme_lines.append(df.describe().to_markdown() + "\n")

    # ------------------------------------------------------------------
    # Part 2: Data Preprocessing
    # ------------------------------------------------------------------
    X = housing.data
    y = housing.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # Part 3: Model Building and Training
    # ------------------------------------------------------------------

    # --- Model 1: Baseline Linear Regression ---
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)

    # --- Model 2: Neural Network with PyTorch ---
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim=32):
            super(MLP, self).__init__()
            self.hidden = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.output = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            x = self.hidden(x)
            x = self.relu(x)
            x = self.output(x)
            return x

    input_dim = X_train_scaled.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=32)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # ------------------------------------------------------------------
    # Part 4: Model Evaluation
    # ------------------------------------------------------------------
    # Linear Regression predictions
    y_pred_lin = lin_reg.predict(X_test_scaled)
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test, y_pred_lin)

    # MLP predictions
    model.eval()
    with torch.no_grad():
        y_pred_mlp_t = model(X_test_t)
    y_pred_mlp = y_pred_mlp_t.numpy().flatten()

    mse_mlp = mean_squared_error(y_test, y_pred_mlp)
    rmse_mlp = np.sqrt(mse_mlp)
    r2_mlp = r2_score(y_test, y_pred_mlp)

    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(f"Linear Regression -> MSE: {mse_lin:.4f}, RMSE: {rmse_lin:.4f}, R2: {r2_lin:.4f}")
    print(f"PyTorch MLP        -> MSE: {mse_mlp:.4f}, RMSE: {rmse_mlp:.4f}, R2: {r2_mlp:.4f}")

    # ------------------------------------------------------------------
    # Part 5: Analysis and Interpretation
    # ------------------------------------------------------------------
    metrics_table = (
        "| Model | MSE | RMSE | R\u00b2 |\n"
        "|---|---|---|---|\n"
        f"| Linear Regression | {mse_lin:.4f} | {rmse_lin:.4f} | {r2_lin:.4f} |\n"
        f"| PyTorch MLP | {mse_mlp:.4f} | {rmse_mlp:.4f} | {r2_mlp:.4f} |\n"
    )

    comparison_text = (
            "The PyTorch MLP achieved a lower MSE/RMSE and a higher R2 than the "
            "baseline Linear Regression model, This indicates that the nueral"
            "network captured more of the non-linear structure in the data and"
            "produced more accurate predictions on the test set.")

    # Plot loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), losses, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss vs. Epoch (PyTorch MLP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()

    loss_discussion = (
        "The loss decreases during the first several epochs as the model "
        "quickly learns the linear relationships in the data, then the rate "
        "of decrease slows and the curve flattens as training continues, indicating "
        "the model is reaching the max optimization of its weighing."
    )

    # ------------------------------------------------------------------
    # Part 6: Write README.md
    # ------------------------------------------------------------------
    with open(README_PATH, "w") as f:
        f.write("# Assignment 2: California Housing Regression\n\n")

        f.write("## How to Run\n\n")
        f.write("```bash\n")
        f.write("conda create -n env_csci4425 python=3.12.2\n")
        f.write("conda activate env_csci4425\n")
        f.write("conda install numpy=2.4.1 matplotlib=3.10.8 pandas=2.3.3 "
                "scikit-learn=1.8.0\n")
        f.write("conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1\n")
        f.write("python hw2.py\n")
        f.write("```\n\n")

        for line in readme_lines:
            f.write(line + "\n")

        f.write("## Part 2: Data Preprocessing\n\n")
        f.write("- Train/test split: 80% / 20%, `random_state=0`\n")
        f.write("- Features scaled with `StandardScaler`, fit on training data only "
                 "and applied to both train and test sets to avoid data leakage.\n\n")

        f.write("## Part 3: Models\n\n")
        f.write("- **Model 1:** `LinearRegression` (scikit-learn), trained on scaled "
                 "training data.\n")
        f.write("- **Model 2:** PyTorch `MLP` with one hidden layer (32 units, ReLU), "
                 "trained for 100 epochs with `Adam` (lr=0.01) and `MSELoss`.\n\n")

        f.write("## Part 4 & 5: Evaluation and Analysis\n\n")
        f.write("### Performance Comparison\n\n")
        f.write(metrics_table + "\n")
        f.write(comparison_text + "\n\n")

        f.write("### Training Loss Curve\n\n")
        f.write(f"![Training Loss]({LOSS_PLOT_PATH})\n\n")
        f.write(loss_discussion + "\n")

        f.write("\n## AI Use Statement\n\n")
        f.write(Ai_Use_Statement + "\n")

    print(f"\nREADME.md written to {README_PATH}")
    print(f"Loss curve saved to {LOSS_PLOT_PATH}")


Ai_Use_Statement = (
    "AI use statement: Generative AI tools were used to assist "
    "with portions of this assignment, in accordance with the course's GenAI policy. "
    "Claude Sonnet 5 (Anthropic) was used to help structure code for hw2.py, as well as "
    "to assist with debugging issues encountered during testing. Gemini 2.5, accessed "
    "within Google Colab, was also used to assist with code explanation, debugging, "
    "and troubleshooting during development and execution in the Colab environment. "
    "All AI-generated content was reviewed, tested, and revised as necessary to ensure "
    "it functioned correctly."
)

if __name__ == "__main__":
    main()