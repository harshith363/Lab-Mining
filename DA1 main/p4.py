import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

file_name = r'C:\Users\harsh\OneDrive\Documents\VIT\FALLSEM 24-25\CBS3007\DA1 main\stall.csv'
df = pd.read_csv(file_name)

print("Harshith 21BBS0163")

frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print("Frequent Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
