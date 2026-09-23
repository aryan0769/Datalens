import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. Dataset Downloader Module
# ==========================================
def download_sample_datasets(data_dir: str = "data") -> str:
    """Downloads sample datasets into the specified directory if they do not exist."""
    os.makedirs(data_dir, exist_ok=True)

    datasets = {
        "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "penguins": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
    }

    primary_file = os.path.join(data_dir, "titanic.csv")

    for name, url in datasets.items():
        file_path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(file_path):
            print(f"[INFO] Downloading {name} dataset from {url}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"[SUCCESS] Saved: {file_path}")
        else:
            print(f"[INFO] File already exists: {file_path}")

    return primary_file


# ==========================================
# 2. DataLens Core EDA & Profiling Engine
# ==========================================
class DataLens:

    def __init__(self, filepath: str, output_dir: str = "output"):
        self.filepath = filepath
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")

        # Create output directories
        os.makedirs(self.plots_dir, exist_ok=True)

        # Load dataset
        self.df = pd.read_csv(filepath)

    def get_summary_table(self) -> pd.DataFrame:
        """Generates column-level data profiling summary."""
        summary = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            missing = self.df[col].isnull().sum()
            missing_pct = round((missing / len(self.df)) * 100, 2)
            unique_cnt = self.df[col].nunique()

            summary.append({
                "Column": col,
                "Type": dtype,
                "Missing Values": missing,
                "Missing (%)": missing_pct,
                "Unique Values": unique_cnt,
            })
        return pd.DataFrame(summary)

    def detect_outliers_iqr(self) -> pd.DataFrame:
        """Calculates outliers using the IQR bounds for numeric features."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_data = []

        for col in num_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            outlier_count = self.df[
                (self.df[col] < lower) | (self.df[col] > upper)
            ].shape[0]

            outlier_data.append({
                "Column": col,
                "Q1 (25%)": round(q1, 2),
                "Q3 (75%)": round(q3, 2),
                "IQR": round(iqr, 2),
                "Outlier Count": outlier_count,
            })
        return pd.DataFrame(outlier_data)

    def generate_visualizations(self) -> list:
        """Saves correlation matrix and distribution plots."""
        saved_plots = []

        # 1. Correlation Heatmap (for numerical features)
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 1:
            plt.figure(figsize=(8, 5))
            corr = self.df[num_cols].corr()
            sns.heatmap(
                corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
            )
            plt.title("Correlation Matrix")
            heatmap_path = os.path.join(
                self.plots_dir, "correlation_heatmap.png"
            )
            plt.tight_layout()
            plt.savefig(heatmap_path)
            plt.close()
            saved_plots.append("plots/correlation_heatmap.png")

        # 2. Histograms for Numerical Columns
        for col in num_cols[:4]:  # Top 4 numerical columns
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col].dropna(), kde=True, color="skyblue")
            plt.title(f"Distribution of {col}")
            plot_path = os.path.join(self.plots_dir, f"dist_{col}.png")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            saved_plots.append(f"plots/dist_{col}.png")

        return saved_plots

    def export_html_report(self):
        """Compiles stats and visuals into a styled standalone HTML file."""
        summary_df = self.get_summary_table()
        outliers_df = self.detect_outliers_iqr()
        plot_paths = self.generate_visualizations()

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DataLens - Automated EDA Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background-color: #f8f9fa; color: #333; }}
                h1, h2 {{ color: #2c3e50; border-bottom: 2px solid #34495e; padding-bottom: 5px; }}
                .meta-box {{ background: #eef2f7; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                th {{ background-color: #34495e; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .img-container {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }}
                .img-card {{ background: white; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                img {{ max-width: 400px; height: auto; display: block; }}
            </style>
        </head>
        <body>
            <h1>DataLens Automated EDA Report</h1>
            <div class="meta-box">
                <p><strong>Input Dataset:</strong> {os.path.basename(self.filepath)}</p>
                <p><strong>Total Rows:</strong> {self.df.shape[0]} | <strong>Total Columns:</strong> {self.df.shape[1]}</p>
            </div>
            
            <h2>1. Dataset Profiling</h2>
            {summary_df.to_html(index=False, classes='table')}
            
            <h2>2. Outlier Analysis (IQR Method)</h2>
            {outliers_df.to_html(index=False, classes='table')}
            
            <h2>3. Visual Analysis</h2>
            <div class="img-container">
        """

        for img in plot_paths:
            html_content += f"""
                <div class="img-card">
                    <img src="{img}" alt="Data Plot">
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        report_path = os.path.join(self.output_dir, "eda_report.html")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(
            f"\n[SUCCESS] EDA Report generated successfully at: {os.path.abspath(report_path)}"
        )


# ==========================================
# 3. Main Execution Workflow
# ==========================================
if __name__ == "__main__":
    # Step 1: Ensure sample dataset exists in data/ directory
    dataset_file = download_sample_datasets(data_dir="data")

    # Step 2: Initialize DataLens and generate HTML report
    lens = DataLens(filepath=dataset_file, output_dir="output")
    lens.export_html_report()