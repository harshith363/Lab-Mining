import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

file_name = r'C:\Users\harsh\OneDrive\Documents\VIT\FALLSEM 24-25\CBS3007\DA1 main\cars.csv'
df = pd.read_csv(file_name)

car = df.groupby('TransactionID')['Item'].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(car).transform(car)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

min_support = 0.4

frequent_itemsets = fpgrowth(
    df_onehot, min_support=min_support, use_colnames=True)

print("Harshith 21BBS0163")
print("Frequent Itemsets:")
print(frequent_itemsets)

min_confidence = 0.7

rules = association_rules(
    frequent_itemsets, metric="confidence", min_threshold=min_confidence)

print("\nAssociation Rules:")
print(rules)
