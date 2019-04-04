
# Updates

- I validated the pairs in Excel (visually) and most of them were in tune with
what we observed from the MBA report.

- I also obtained the following (top-20) constraints along with their 0-1 values.
They follow the same pattern like the outputs from mba. However there are
two pairs which have a strong negative dependence: 
    - Essential Hypertension, Other Upper Respiratory Infections
    - Diabetes Mellitus, Other Upper Respiratory Infections

- I intend to run it for the across the 254 diseases and see what top pairs 
come out there but I dont know how to interpret it since we will anyways be
focusing on a subset of the full disease set (or not???)

Here are the pairs:

('Lipid Metabolism Disorder', 'Essential Hypertension') : (1, 1)


('Diabetes Mellitus', 'Essential Hypertension') : (1, 1)


('Diabetes Mellitus', 'Lipid Metabolism Disorder') : (1, 1)


('Acute Myocardial Infarction', 'Coronary Atherosclerosis and Other Heart Disease') : (1, 1)


('Essential Hypertension', 'Coronary Atherosclerosis and Other Heart Disease') : (1, 1)


('Lipid Metabolism Disorder', 'Coronary Atherosclerosis and Other Heart Disease') : (1, 1)


('Essential Hypertension', 'Non Traumatic Joint Disorders') : (1, 1)


('Essential Hypertension', 'Osteoarthritis') : (1, 1)


('Essential Hypertension', 'Acute Myocardial Infarction') : (1, 1)


('Essential Hypertension', 'Rheumatoid Arthritis and Related Disease') : (1, 1)


('Essential Hypertension', 'Other Upper Respiratory Infections') : (0, 1)


('Lipid Metabolism Disorder', 'Non Traumatic Joint Disorders') : (1, 1)


('Diabetes Mellitus', 'Coronary Atherosclerosis and Other Heart Disease') : (1, 1)


('Non Traumatic Joint Disorders', 'Spondylosis; Intervertebral Disc Disorders; Other Back Problems') : (1, 1)


('Non Traumatic Joint Disorders', 'Other Connective Tissue Disorders') : (1, 1)


('Lipid Metabolism Disorder', 'Acute Myocardial Infarction') : (1, 1)


('Diabetes Mellitus', 'Other Upper Respiratory Infections') : (1, 0)


('Lipid Metabolism Disorder', 'Osteoarthritis') : (1, 1)


('Chronic Obstructive Pulmonary Disease and Bronchiectasis', 'Asthma') : (1, 1)


('Spondylosis; Intervertebral Disc Disorders; Other Back Problems', 'Other Connective Tissue Disorders') : (1, 1)




## Some questions

I noticed that there were some numerical underflow issues when computing the 
L-measure L(X;Y) between two diseases. Perhaps this is due to high sparsity of the 
problem when computing the constraints on the entire data set and the relatively
low values for mutual info themselves (order of $10^{-7}$).
I was wondering
how close the mutual information I(X;Y) will mirror L(X;Y) atleast for the 
binary case (intutively??). 
Both of them lie between 0 and 1 (0 indicates independence). Using
I(X;Y) values is much much faster (code runtime wise).


