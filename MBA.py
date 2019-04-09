# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:28:34 2019

@author: Prasanna
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

class MBA():
    """Class: takes in a cleaned data set in a csv format in which diseases are features with
       that take binary values..
       Attributes:
           bask: This is a csv file used on processed and binarized disease data
           min_support: The minimum support to filter frequent item sets of diseases from the MLXtend library. Default is 0.004
           rules: This is used to further filter or rank item sets. Default is lift. Other args are listed as follows"""

    def __init__(self, cleaneddata, min_support=0.04, metric='lift', min_threshold=0.001):
        self.cleaneddata=pd.read_excel(directory)
        self.min_support=min_support
        self.metric=metric
        self.min_threshold=min_threshold

    print("Performing the Market Basket Analysis")
    def marketbasket(cleaneddata):
        frequent_itemsets = apriori(basket_sets, min_support=0.004, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.001)
        rules["antecedant_len"] = rules["antecedants"].apply(lambda x: len(x))
        rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))
        indices_two=list(rules.loc[(rules['antecedant_len'] == 1) & (rules['consequent_len'] == 1)].sort_values(by=['lift'], ascending=False).index.values)
        indices_three=list(rules.loc[((rules['antecedant_len'] == 2) & (rules['consequent_len'] == 1)) | ((rules['antecedant_len'] == 1) & (rules['consequent_len'] == 2)) ].sort_values(by=['lift'], ascending=False).index.values)
        indices_four=list(rules.loc[((rules['antecedant_len'] == 2) & (rules['consequent_len'] == 2)) | ((rules['antecedant_len'] == 1) & (rules['consequent_len'] == 3)) | ((rules['antecedant_len'] == 3) & (rules['consequent_len'] == 1)) ].sort_values(by=['lift'], ascending=False).index.values)
        """Declare empty sets to store itemsets of sizes two, three and four"""
        sets_two=set([])
        sets_three=set([])
        sets_four=set([])
        val_two=(1,1)
        val_three=(1,1,1)
        val_four=(1,1,1,1)
        two_way_dict={}
        three_way_dict={}
        four_way_dict={}
        print("Loading the data into a dictionary")
        """Add frozen sets of pairs, triplets and quadruplets to declared empty sets"""
        for itwo in indices_two:
            sets_two.add(frozenset().union(rules.iloc[i2]['antecedants'],rules.iloc[itwo]['consequents']))
        for ithree in indices_three:
            sets_three.add(frozenset().union(rules.iloc[i3]['antecedants'],rules.iloc[ithree]['consequents']))
        for ifour in indices_four:
            sets_four.add(frozenset().union(rules.iloc[i4]['antecedants'],rules.iloc[ifour]['consequents']))
        """Output two way, three way and four way dictionaries as input for optimization"""
        for stwo in sets_two:
            two_way_dict[tuple(stwo)]=val_two
        for sthree in sets_three:
            three_way_dict[tuple(sthree)]=val_three
        for sfour in sets_four:
            four_way_dict[tuple(sfour)]=val_four
        return two_way_dict, three_way_dict, four_way_dict

        
