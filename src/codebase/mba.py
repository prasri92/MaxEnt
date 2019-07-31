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
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
    indices_two=list(rules.loc[(rules['antecedent_len'] == 1) & (rules['consequent_len'] == 1)].sort_values(by=['lift'], ascending=False).index.values)
    indices_three=list(rules.loc[((rules['antecedent_len'] == 2) & (rules['consequent_len'] == 1)) | ((rules['antecedent_len'] == 1) & (rules['consequent_len'] == 2)) ].sort_values(by=['lift'], ascending=False).index.values)
    indices_four=list(rules.loc[((rules['antecedent_len'] == 2) & (rules['consequent_len'] == 2)) | ((rules['antecedent_len'] == 1) & (rules['consequent_len'] == 3)) | ((rules['antecedent_len'] == 3) & (rules['consequent_len'] == 1)) ].sort_values(by=['lift'], ascending=False).index.values)
    """Declare empty sets to store itemsets of sizes two, three and four"""
    sets_two = set()
    sets_three = set()
    sets_four = set()
    val_two=(1,1)
    val_three=(1,1,1)
    val_four=(1,1,1,1)
    two_way_dict={}
    three_way_dict={}
    four_way_dict={}
    # print("Loading the data into a dictionary")
    """Add frozen sets of pairs, triplets and quadruplets to declared empty sets"""
    for itwo in indices_two:
        ltwo = []
        a = rules.iloc[itwo]['antecedents']
        b = rules.iloc[itwo]['consequents']
        sets_two.add(a.union(b))
    ltwo=[]
    for i in sets_two:
        tmp = []
        for j in i:
            tmp.append(int(float(j)))
        ltwo.append(tmp)

    for ithree in indices_three:
        lthree = []
        a = rules.iloc[ithree]['antecedents']
        b = rules.iloc[ithree]['consequents']
        sets_three.add(a.union(b))
    lthree=[]
    for i in sets_three:
        tmp = []
        for j in i:
            tmp.append(int(float(j)))
        lthree.append(tmp)

    for ifour in indices_four:
        lfour = []
        a = rules.iloc[ifour]['antecedents']
        b = rules.iloc[ifour]['consequents']
        sets_four.add(a.union(b))
    
    lfour = []
    for i in sets_four:
        tmp = []
        for j in i:
            tmp.append(int(float(j)))
        lfour.append(tmp)
    
    """Output two way, three way and four way dictionaries as input for optimization"""
    for stwo in ltwo:
        two_way_dict[tuple(stwo)]=val_two
    for sthree in lthree:
        three_way_dict[tuple(sthree)]=val_three
    for sfour in lfour:
        four_way_dict[tuple(sfour)]=val_four
    
    return two_way_dict, three_way_dict, four_way_dict