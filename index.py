import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import numpy as np

# Set up output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Debug: Show working dir and data files
print("CWD:", os.getcwd())
print("Files in data/:", os.listdir("data"))

# Load CSV (assume comma separator)
df = pd.read_csv("data/twcs.csv", sep=',')

# Show a sample of the data
print(df.head())

# --- 1. Text Length Analysis ---
if 'text' in df.columns:
    df['text_length'] = df['text'].astype(str).str.len()
    sns.histplot(df['text_length'], bins=30)
    plt.title("Distribution of Message Lengths")
    plt.xlabel("Message Length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "text_length_distribution.png"))
    plt.close()
else:
    print("Column 'text' not found!")

# --- 2. Inbound/Outbound Distribution ---
if 'inbound' in df.columns:
    df['inbound'] = df['inbound'].astype(str).str.lower().map({'true': True, 'false': False})
    sns.countplot(data=df, x='inbound')
    plt.title("Inbound vs Outbound Messages")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inbound_outbound_distribution.png"))
    plt.close()
else:
    print("Column 'inbound' not found!")

# --- 3. Author Activity ---
if 'author_id' in df.columns:
    top_authors = df['author_id'].value_counts().head(10)
    top_authors.plot(kind='bar')
    plt.title("Top 10 Most Active Authors")
    plt.ylabel("Number of Messages")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_10_authors.png"))
    plt.close()
else:
    print("Column 'author_id' not found!")

# --- 4. Time Patterns ---
if 'created_at' in df.columns:
    # Parse Twitter datetime: Tue Oct 31 22:10:47 +0000 2017
    df['created_at'] = pd.to_datetime(
        df['created_at'],
        format="%a %b %d %H:%M:%S %z %Y",
        errors='coerce'
    )
    df['hour'] = df['created_at'].dt.hour
    sns.histplot(df['hour'].dropna(), bins=24)
    plt.title("Tickets by Hour of Day")
    plt.xlabel("Hour")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tickets_by_hour.png"))
    plt.close()
else:
    print("Column 'created_at' not found!")

# --- 5. Data Quality Checks ---
print("\nData Quality Checks:")
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate tweet_ids:", df['tweet_id'].duplicated().sum())

print(f"\nAll visualizations are saved in the '{output_dir}' folder.")

# --- 6. Formal Hypothesis Tests ---
print("\n===== Formal Hypothesis Tests =====")

# --- 1. Inbound vs Outbound Message Length ---
if {'inbound', 'text_length'}.issubset(df.columns):
    inbound_lengths = df[df['inbound'] == True]['text_length'].dropna()
    outbound_lengths = df[df['inbound'] == False]['text_length'].dropna()

    # Two-sample t-test (Welch’s t-test, no equal variance assumption)
    t_stat, p_value = stats.ttest_ind(inbound_lengths, outbound_lengths, equal_var=False)

    print(f"\n[Inbound vs Outbound Message Length]")
    print(f"Mean inbound: {inbound_lengths.mean():.2f}, outbound: {outbound_lengths.mean():.2f}")
    print(f"t-statistic = {t_stat:.2f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Reject H0. Message lengths differ significantly.")
    else:
        print("Result: Fail to reject H0. No significant difference in message lengths.")
else:
    print("Cannot test inbound vs outbound message length (missing columns).")

# --- 2. Are tickets uniformly distributed across hours? ---
if 'hour' in df.columns:
    # Ensure all 24 hours are represented, even if zero
    hour_counts = df['hour'].dropna().astype(int).value_counts().sort_index()
    hour_counts = hour_counts.reindex(range(24), fill_value=0)
    observed = hour_counts.values
    expected = np.full(24, observed.sum() / 24)  # Each hour gets equal expected count, matching sum

    # Chi-square goodness-of-fit test
    chi2_stat, p_value = stats.chisquare(observed, expected)
    print(f"\n[Uniform Distribution of Tickets Across Hours]")
    print(f"Chi2 statistic = {chi2_stat:.2f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("Result: Reject H0. Tickets are not uniformly distributed across hours (peaks exist).")
    else:
        print("Result: Fail to reject H0. No significant hourly peaks.")
else:
    print("Cannot test uniformity across hours (missing 'hour' column).")
