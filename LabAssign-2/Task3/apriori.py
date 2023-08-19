#%%
import numpy as np
from numpy import genfromtxt
from collections import Counter
from itertools import combinations
from copy import deepcopy

# absolute support threshold
abs_threshold = 18

# Loading the data and reading in books
data = open("books.txt","r")
all_books = []
samples = [] # Lines from data

# Getting all books from books.txt
for sample in data.read().splitlines():
    samples.append(sample.split(";"))
    for book in sample.split(";"):
        all_books.append(book)

# counts the elements' frequency
unique_books = list(Counter(all_books).keys())
support = list(Counter(all_books).values()) 

L = [] # all 1-frequent books with abs_support

# 1-frequent books with their support
with open('oneItems.txt', 'w') as oneItems:
    
    for book, support_book in zip(unique_books, support):
        if support_book > abs_threshold:
            L.append([book])
            oneItems.write(str(support_book)+":"+str(book)+"\n")
        
oneItems.close()
data.close()

""" all frequent itemsets with minimal support (excluding the 
    1-frequent items, since they can be found in oneItems.txt) """

patterns = open('patterns.txt', 'w')

# Stop if no more frequent items with given length 
while len(L) != 0:

    ### Building candidates
    books = [] 
    C = [] # candidate itemsets (un-pruned)
    for book in unique_books:
        books.append(book)
        for itemset in L:
            # Checks if the itemset contains the book
            # if not then add it to the itemset
            if not any(elem in books for elem in itemset):
                # deepcopy to not overwrite the itemset
                copy = deepcopy(itemset)
                copy.append(book)
                C.append(copy)

    ### Pruning
    # Removing any set below abs threshold
    # and recording the respective supports
    L = []
    supports = []
    for itemset in C:
        support = 0
        for sample in samples:
            
            # Calculates the abs. support of the itemset
            if all(elem in sample for elem in itemset):
                support += 1
        
        # Frequent itemsets
        if support > abs_threshold:
            L.append(itemset)
            supports.append(support)
        
            # Write to file
            patterns.write(str(support)+':')
            for num, item in enumerate(itemset,0):
                if num == len(itemset) - 1:
                    patterns.write(item+'\n')
                else:
                    patterns.write(item+';')

patterns.close()
# %%
""" 3c was not completed. """