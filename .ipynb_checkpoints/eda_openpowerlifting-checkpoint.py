"""
OpenPowerlifting Meet Results - Exploratory Data Analysis
Student: Naveen
Dataset: openpowerlifting.csv (place this file in the same folder as this script,
         or update the path in pd.read_csv() below)

Run with:  python eda_openpowerlifting.py
(Charts will pop up one by one via plt.show() - close each window to continue,
 or run this file's contents inside Jupyter for inline charts.)
"""

# # OpenPowerlifting Meet Results — Exploratory Data Analysis
# **Student:** Bhanu
# **Dataset:** OpenPowerlifting competition results (`openpowerlifting.csv`)
# **Presentation Date:** 17 September 2026
# This notebook performs a complete EDA project following the assignment guidelines: business understanding, data cleaning, univariate/bivariate/multivariate analysis, pivot tables, groupby analysis, visualizations, correlation analysis, and business insights.

# ## 1. Business Understanding
# **Industry / Domain:** Sports & Fitness — competitive powerlifting.
# **Business Problem:** A powerlifting federation / gym chain wants to understand athlete performance patterns across its sanctioned meets, so it can design better coaching programs, set fair weight-class and equipment rules, and identify what drives top‑level performance for scouting and marketing purposes.
# **Project Objective:** Analyze historical meet results to understand how sex, bodyweight, age, equipment category and weight class relate to lifting performance (Squat, Bench, Deadlift, Total, and the bodyweight‑adjusted Wilks score), and surface actionable insights for the federation's coaching and event‑planning teams.
# **Key Performance Indicators (KPIs):**
# - **Average Total (kg)** lifted — overall strength benchmark
# - **Average Wilks Score** — bodyweight‑normalized strength benchmark (fair comparison across weight classes)
# - **Completion / No‑Lift Rate** — share of athletes who failed to record a valid Best lift
# - **Participation Mix** — share of athletes by Sex and Equipment category
# - **Disqualification (DQ) Rate** — share of entries disqualified

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 20)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)

df = pd.read_csv('openpowerlifting__1_.xls')
print("Dataset loaded.")
print("Shape (rows, columns):", df.shape)

df.head()
print(df.head().to_string())

# ## 2. Data Cleaning

# ### 2.1 Dataset Shape Analysis

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# ### 2.2 Data Type Analysis

print(df.dtypes)

# ### 2.3 Missing Value Analysis

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_report = missing_report[missing_report['missing_count'] > 0].sort_values('missing_pct', ascending=False)
print(missing_report)

# ### 2.4 Duplicate Record Analysis

dup_count = df.duplicated().sum()
print(f"Fully duplicated rows: {dup_count}")

# **Observation:** `Squat4Kg`, `Bench4Kg`, and `Deadlift4Kg` (4th/extra attempt columns) are missing for ~99% of rows — they only apply to record-attempt lifts. `Age`, `Division`, and the raw attempt columns have substantial missingness; `BestSquatKg`, `BestBenchKg`, `BestDeadliftKg` have missing values where an athlete no‑lifted that movement (bombed out).
# **Insight:** Missingness here is largely *structural* (not random) — it reflects the rules of competition (not every athlete attempts every lift, few attempt a 4th record lift).
# **Business Impact:** Any KPI based on Best lift columns must account for genuine no‑lifts (competition outcome) separately from data‑quality gaps, otherwise average performance will be biased upward.

# ### 2.5 Handling Missing Values, Removing Unnecessary Columns, Removing Duplicates

# Drop the near-empty 4th-attempt columns - not needed for this analysis (>99% missing, record-attempt only)
df_clean = df.drop(columns=['Squat4Kg', 'Bench4Kg', 'Deadlift4Kg'])

# Remove fully duplicated rows
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Removed {before - len(df_clean)} duplicate rows")

# Keep NaNs in BestSquatKg/BestBenchKg/BestDeadliftKg (they are genuine no-lifts, not junk)
# but drop rows with no usable performance data at all (no Total and no Best lift of any kind)
before = len(df_clean)
df_clean = df_clean.dropna(subset=['TotalKg', 'BestSquatKg', 'BestBenchKg', 'BestDeadliftKg'], how='all')
print(f"Removed {before - len(df_clean)} rows with zero recorded lift data")

# Age and BodyweightKg: keep as-is (NaN preserved), used only where relevant per analysis
print("Cleaned shape:", df_clean.shape)

# ### 2.6 Data Consistency Checks

print("Sex categories:", sorted(df_clean['Sex'].dropna().unique()))
print("Equipment categories:", sorted(df_clean['Equipment'].dropna().unique()))
print("Age range:", df_clean['Age'].min(), "-", df_clean['Age'].max())
print("BodyweightKg range:", round(df_clean['BodyweightKg'].min(),1), "-", round(df_clean['BodyweightKg'].max(),1))
print("Negative/zero TotalKg rows:", (df_clean['TotalKg'] <= 0).sum())
print("Place value sample:", df_clean['Place'].dropna().unique()[:15])

# **Observation:** Sex (F/M) and Equipment (5 categories) are clean, consistent categories. `Place` mixes numeric ranks with codes `DQ` (disqualified), `G` (guest lifter — not competing for place), and `NS` (no‑show). Age and Bodyweight ranges are plausible (youth to masters divisions).
# **Insight:** These non‑numeric `Place` codes must be treated as categorical outcomes, not filtered out as errors — they carry business meaning (DQ rate, no‑show rate).

# ## 3. Exploratory Data Analysis (EDA)

# ### 3.1 Univariate Analysis

fig, axes = plt.subplots(1, 2, figsize=(13,5))
sns.histplot(df_clean['Age'].dropna(), bins=40, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Distribution of Athlete Age')
axes[0].set_xlabel('Age (years)')

sns.histplot(df_clean['BodyweightKg'].dropna(), bins=40, kde=True, ax=axes[1], color='indianred')
axes[1].set_title('Distribution of Bodyweight (Kg)')
axes[1].set_xlabel('Bodyweight (Kg)')
plt.tight_layout()
plt.show()

print(df_clean[['Age','BodyweightKg','TotalKg','Wilks']].describe().round(2))

fig, ax = plt.subplots(1, 2, figsize=(13,5))
df_clean['Sex'].value_counts().plot(kind='bar', ax=ax[0], color=['#e377c2','#1f77b4'])
ax[0].set_title('Athlete Count by Sex')
ax[0].set_xlabel('Sex'); ax[0].set_ylabel('Count')

df_clean['Equipment'].value_counts().plot(kind='bar', ax=ax[1], color='seagreen')
ax[1].set_title('Athlete Entries by Equipment Category')
ax[1].set_xlabel('Equipment'); ax[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

# **Observation:** The male athlete population is noticeably larger than the female population in this dataset. `Raw` and `Single-ply` are the dominant equipment categories.
# **Insight:** Sex and Equipment are highly imbalanced categorical fields — any cross-category comparison should use normalized metrics (averages/rates), not raw counts.
# **Business Impact:** Federation marketing and outreach could target growing female and Raw-division participation, which appear to be the fastest-growing but still smaller segments.

# ### 3.2 Bivariate Analysis

fig, ax = plt.subplots(figsize=(7,5))
sns.boxplot(data=df_clean, x='Sex', y='TotalKg', ax=ax, palette=['#e377c2','#1f77b4'])
ax.set_title('Total Lifted (Kg) by Sex')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9,5))
sns.boxplot(data=df_clean, x='Equipment', y='TotalKg', ax=ax, palette='Set2')
ax.set_title('Total Lifted (Kg) by Equipment Category')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

sample = df_clean.sample(min(20000, len(df_clean)), random_state=42)
fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=sample, x='BodyweightKg', y='TotalKg', hue='Sex', alpha=0.3, s=15, ax=ax)
ax.set_title('Bodyweight vs Total Lifted (sampled)')
plt.tight_layout()
plt.show()

# **Observation:** Male athletes and equipped lifters (Single-ply/Multi-ply, which allow supportive gear) post noticeably higher Total Kg than female or Raw athletes. Total lifted rises with bodyweight but with diminishing/plateauing returns at the high end.
# **Insight:** Raw comparisons of Total Kg conflate sex, equipment, and bodyweight effects — this is exactly why the sport uses the bodyweight-normalized **Wilks score** for fair cross-category ranking.
# **Business Impact:** Federation award/scouting decisions based on raw Total alone would systematically favor heavier, equipped male lifters — Wilks-based leaderboards are needed for fair recognition across categories.

# ### 3.3 Multivariate Analysis

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(data=df_clean, x='Equipment', y='Wilks', hue='Sex', ax=ax, palette=['#e377c2','#1f77b4'])
ax.set_title('Wilks Score by Equipment and Sex')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

sample2 = df_clean.dropna(subset=['Age']).sample(min(20000, df_clean['Age'].notna().sum()), random_state=42)
fig, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=sample2, x='Age', y='Wilks', hue='Sex', alpha=0.25, s=15, ax=ax)
ax.set_title('Age vs Wilks Score by Sex')
plt.tight_layout()
plt.show()

# **Observation:** Even on the normalized Wilks score, Equipment and Sex still create visible group differences, and Wilks score peaks in the 20s–30s age range before declining with age.
# **Insight:** Peak competitive strength (Wilks-adjusted) is concentrated in early adulthood, consistent with known strength-training physiology; this is a multivariate effect only visible once Age, Sex and Equipment are viewed together.
# **Business Impact:** Coaching programs and athlete development pipelines should prioritize the 20–35 age band for elite scouting, while masters (40+) programs should be benchmarked against age-adjusted, not open-division, standards.

# ## 4. Pivot Table Analysis

pivot1 = pd.pivot_table(df_clean, values='TotalKg', index='Sex', columns='Equipment', aggfunc='mean').round(1)
print("Average Total Kg by Sex x Equipment:")
print(pivot1)

pivot2 = pd.pivot_table(df_clean, values='Wilks', index='Sex', columns='Equipment', aggfunc='mean').round(1)
print("Average Wilks Score by Sex x Equipment:")
print(pivot2)

df_clean['AgeGroup'] = pd.cut(df_clean['Age'], bins=[0,18,23,29,39,49,59,120],
                              labels=['Teen(<=18)','19-23','24-29','30-39','40-49','50-59','60+'])
pivot3 = pd.pivot_table(df_clean, values='Wilks', index='AgeGroup', columns='Sex', aggfunc='mean', observed=False).round(1)
print("Average Wilks Score by Age Group x Sex:")
print(pivot3)

# **Observation:** Equipped categories (Single-ply, Multi-ply) show higher average Total Kg than Raw/Wraps/Straps across both sexes, and this gap persists (though shrinks) even on the normalized Wilks score. The 24–29 and 30–39 age groups post the highest average Wilks scores for both sexes.
# **Insight:** Equipment provides a measurable mechanical advantage beyond what bodyweight-normalization alone corrects for, and competitive strength peaks in the late-20s-to-30s window.
# **Business Impact:** The federation should keep equipment categories strictly separated in rankings and awards (already common practice) and can use the 24–39 age-group benchmark as the target standard for "open division" coaching programs.

# ## 5. GroupBy Analysis

equip_summary = df_clean.groupby('Equipment', observed=True).agg(
    entries=('Name','count'),
    avg_total=('TotalKg','mean'),
    avg_wilks=('Wilks','mean')
).round(1).sort_values('entries', ascending=False)
print("Category-wise (Equipment) performance summary:")
print(equip_summary)

top_meets = df_clean.groupby('MeetID').agg(
    entries=('Name','count'),
    avg_total=('TotalKg','mean')
).sort_values('entries', ascending=False).head(10).round(1)
print("Top 10 meets by number of entries:")
print(top_meets)

dq_rate = (df_clean['Place'] == 'DQ').mean() * 100
print(f"Overall disqualification (DQ) rate: {dq_rate:.2f}%")

dq_by_equip = df_clean.groupby('Equipment', observed=True)['Place'].apply(lambda s: (s=='DQ').mean()*100).round(2)
print("\nDQ rate (%) by Equipment:")
print(dq_by_equip.sort_values(ascending=False))

# **Observation:** `Raw` has by far the most entries, reflecting its growing popularity as unequipped lifting. A handful of meets account for a disproportionate share of total entries. DQ rates vary meaningfully by equipment category.
# **Insight:** Participation is concentrated in a small number of large meets and in the Raw category — the federation's growth is being driven primarily by unequipped/Raw competition.
# **Business Impact:** Event planning and resource allocation (judging staff, equipment checks, venue size) should prioritize Raw divisions and the small set of high-turnout meets; equipment categories with higher DQ rates may need clearer rule communication or pre-meet equipment checks.

# ## 6. Data Visualization

fig, ax = plt.subplots(figsize=(6,6))
df_clean['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#1f77b4','#e377c2'])
ax.set_ylabel('')
ax.set_title('Share of Entries by Sex')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8,5))
equip_summary['avg_total'].plot(kind='bar', ax=ax, color='teal')
ax.set_title('Average Total Kg by Equipment Category')
ax.set_ylabel('Average Total (Kg)')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# **Observation:** Multi-ply and Single-ply lead on raw average Total, while Raw and Wraps sit lower — matching the earlier pivot-table finding.
# **Insight:** The bar chart confirms the equipment effect is consistent and large enough to matter for any performance ranking.
# **Business Impact:** Any "strongest lifter" marketing claim should always state the equipment category, or it will misrepresent relative achievement.

# ## 7. Correlation Analysis

numeric_cols = ['Age','BodyweightKg','BestSquatKg','BestBenchKg','BestDeadliftKg','TotalKg','Wilks']
corr = df_clean[numeric_cols].corr().round(2)
print(corr)

fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix — Numeric Performance Features')
plt.tight_layout()
plt.show()

# **Observation:** `BestSquatKg`, `BestBenchKg`, `BestDeadliftKg` and `TotalKg` are all very strongly positively correlated with each other (as expected — Total is their sum). `BodyweightKg` correlates positively with raw lift numbers but far more weakly with `Wilks` (by design, since Wilks removes the bodyweight effect). `Age` has only a weak correlation with performance.
# **Insight:** The three individual lifts move together strongly, so Total Kg is a reliable single proxy for overall strength, while Wilks is the right metric whenever bodyweight needs to be controlled for (e.g., cross-weight-class comparisons).
# **Business Impact:** For fair "pound-for-pound" rankings, award decisions and athlete-of-the-year selections, the federation should standardize on Wilks, not raw Total, as the primary KPI.

# ## 8. Problem-Solving Approach
# **Problem:** The federation lacked a clear, data-driven view of how sex, equipment, age, and bodyweight drive competitive powerlifting performance, making it hard to run fair rankings, target coaching investment, and plan events.
# **Approach:** The 386K-row OpenPowerlifting results dataset was cleaned (structural missing values handled, duplicates removed, inconsistent codes documented), then examined through univariate, bivariate, multivariate, pivot-table and groupby analyses, visualized across standard chart types, and checked for feature correlation.
# **How the data helps:** The analysis isolates the size of the equipment effect, the shape of the age-performance curve, the reliability of Wilks vs raw Total as a KPI, and where participation and disqualification risk concentrate — turning raw meet results into decisions the federation can act on (which KPI to standardize on, which age band to target for scouting, which categories need clearer rules, where to focus event resources).

# ## 9. Business Insights & Recommendations (Summary)
# | # | Insight | Recommendation |
# |---|---------|-----------------|
# | 1 | Equipment (Single-ply/Multi-ply) gives a measurable performance advantage over Raw, even after Wilks normalization | Always report rankings within equipment category, never mixed |
# | 2 | Wilks score is only weakly correlated with bodyweight and age, while Total Kg is strongly driven by both | Standardize official "best lifter" awards on Wilks, not Total |
# | 3 | Peak Wilks performance occurs in the 24–39 age band | Focus elite scouting and athlete development pipelines on this age range; benchmark masters athletes on age-adjusted standards |
# | 4 | Participation is dominated by Raw equipment and a small set of large meets | Prioritize event resources (judges, equipment checks, venue capacity) toward Raw divisions and top meets |
# | 5 | DQ rates differ meaningfully by equipment category | Investigate rule clarity / pre-meet equipment checks for higher-DQ categories |
# | 6 | Male participation and Total Kg outnumber/outpace female figures | Target outreach and marketing toward growing female participation |
# ## 10. Conclusion
# This EDA converted a large, real-world powerlifting results dataset into a clear picture of how sex, equipment, age, and bodyweight shape competitive performance, and produced concrete, KPI-linked recommendations for the federation's ranking policy, coaching focus, and event planning.
