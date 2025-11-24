import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import ttest_ind
import numpy as np

#1
df = pd.read_csv('26.1.csv')
#2
print(df.head())

#3
df.boxplot(column=["A", "B"])
plt.title("Коробкові діаграми для варіантів A і B")
plt.ylabel("Значення")
plt.show()

results = df.agg(["mean", "median"]).T
results.columns = ["Середнє", "Медіана"]

print(results)

#4
combined = pd.concat([df["A"], df["B"]], ignore_index=True)

observed_diff = df["B"].mean() - df["A"].mean()

R = 200
permutation_diffs = []

for i in range(R):
    shuffled = combined.sample(frac=1, random_state=None).reset_index(drop=True)
    new_A = shuffled.iloc[:len(df["A"])]
    new_B = shuffled.iloc[len(df["A"]):]
    diff = new_B.mean() - new_A.mean()
    permutation_diffs.append(diff)

p_value = (abs(pd.Series(permutation_diffs)) >= abs(observed_diff)).mean()

alpha = 0.05

if p_value < alpha:
    conclusion = "Різниця статистично значуща"
else:
    conclusion = "Різниця не є статистично значущою"

print(f"Спостережувана різниця середніх: {observed_diff:.3f}")
print(f"p-значення: {p_value:.4f}")
print(f"Висновок: {conclusion}")

#5
T_test = ttest_ind(df["A"], df["B"], equal_var=False)

t_stat = T_test.statistic
p_value_ttest = T_test.pvalue

if p_value_ttest < alpha:
    conclusion_ttest = "Різниця статистично значуща"
else:
    conclusion_ttest = "Різниця не є статистично значущою"
print(f"Перестановочний тест: p = {p_value:.5f} → {conclusion}")
print(f"t-тест:             p = {p_value_ttest:.5f}, t = {t_stat:.3f} → {conclusion_ttest}")
