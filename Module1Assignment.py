import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("synthetic_student_performance.csv")

subset_cols = ["StudentID", "Extracurricular", "Sports", "Music", "Volunteering", "GPA", "GradeClass"]

subset = df[subset_cols].copy()

subset["InvolvementScore"] = (subset["Extracurricular"] + subset["Sports"] + subset["Music"] + subset["Volunteering"])

print("\n Summary Statistics")
print(subset.describe().round(2))

print("\n Correlation: GPA vs InvolvementScore")
print(subset[["InvolvementScore", "GPA"]].corr().round(2))

print("\n Average GPA by InvolvementScore")
print(subset.groupby("InvolvementScore")["GPA"].mean().round(2))
