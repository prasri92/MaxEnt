"""
mlxtend version = 0.15.0
python = 3.7.3
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def marketbasket(cleaneddata, support):
    '''
    Function to use market basket analysis to return 2 way, 3 way and 4 way constraints. 
    Input: 
        Data vectors from synthetically generated data 
        Support: Support for the apriori algorithm
    Returns: 
        Dictionary containing 2 way, 3 way and 4 way constraints 
    '''
    frequent_itemsets = apriori(cleaneddata, min_support=support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.002)
    #rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    #rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
    rules["itemsets"]=rules.apply(lambda x: x["antecedents"].union(x["consequents"]), axis=1)
    rules["itemsets"]=rules["itemsets"].apply(lambda x: [int(y) for y in x])
    rules["itemsets"]=rules["itemsets"].apply(lambda x: tuple(sorted(x)))
    rules["itemsets_len"]=rules["itemsets"].apply(lambda x: len(x))
    rules=rules.drop_duplicates(subset="itemsets")
    rules=rules.drop(['antecedents', 'consequents', 'antecedent support', 'consequent support', 'confidence', 'lift', 'leverage', 'conviction'], axis=1)
    rules2=rules["itemsets"][rules['itemsets_len']==2]
    rules3=rules["itemsets"][rules['itemsets_len']==3]
    rules4=rules["itemsets"][rules['itemsets_len']==4]
    supportdict=dict(zip(rules["itemsets"], rules["support"]))
    # print("Support dict:", supportdict)
    twowaydict=dict.fromkeys(rules2, (1, 1))
    threewaydict=dict.fromkeys(rules3, (1, 1, 1))
    fourwaydict=dict.fromkeys(rules4, (1, 1, 1, 1))
    
    return supportdict, twowaydict, threewaydict, fourwaydict
