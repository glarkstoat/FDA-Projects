#%%
# Import Packages
import csv
from collections import defaultdict
from itertools import combinations


itemSets = list()   # List of the different item sets
itemSet = set() # Set for the items in the data

# Open the file
with open('books.txt', 'r') as f:

    # Initialize csv reader with ; as seperator
    reader = csv.reader(f, delimiter=';')
    for line in reader:
        # Make a list of the current line
        line = list(line)
        # Extract the item set from the current line
        record = set(line)

        for item in record:
            # Add the items from the current line to the 
            # set of all items
            itemSet.add(frozenset([item]))
        
        # Append the current line as new list to the list of item sets
        itemSets.append(record)
    # Close the file
    f.close()


def aboveMinSup(itemSet, itemSets, minSup, globalItemSet, globalSupports):

    # Helper Function to calculate the supports and generate the frequent item sets
    currentSupports = list() # List of supports for current level of transaction tree
    freqItemSet = set() # Set of frequent item sets in the current level of transaction tree
    localItemSet = defaultdict(int) # Dictionary of the local item sets for the current level

    for item in itemSet:
        for itemSet in itemSets:
            if item.issubset(itemSet):
                # If the current item is a subset of the item set
                # Then add +1 to absolute support to the global and
                # local item set
                globalItemSet[item] += 1
                localItemSet[item] += 1

    for item, supCount in localItemSet.items():
        # Calculate relative support
        support = float(supCount / len(itemSets))

        if support >= minSup:
            # If relative support is greater than the minSup parameter
            # add the absolute support of the current item and add
            # the current item to the frequent item set
            currentSupports.append(supCount)
            freqItemSet.add(item)
    
    # Append the list of absolute supports to the global support list
    globalSupports.append(currentSupports)

    # Return the frequent item set for the current layer to the 
    # frequent item dictionary
    return freqItemSet


def getUnion(itemSet, length):
    # Helper function to create a set of the union for the current level on the transaction tree
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def pruning(candidateSet, prevFreqSet, length):

    # Helper function for the pruning step
    tempCandidateSet = candidateSet.copy() # Copy the candidate set

    for item in candidateSet:
        # Create subsets for the current level from combinations
        # of the current item in the candidate set
        subsets = combinations(item, length)
        
        for subset in subsets:
            # Check if the candidate subset is part of the previous layer
            # if not, then remove it from the candidate set and go to the 
            # next subset (frozenset to not mess with the entries)
            if frozenset(subset) not in prevFreqSet:
                tempCandidateSet.remove(item)
                break

    # Return the pruned candidate set
    return tempCandidateSet


def writeFiles(globalFreqItglobalFreqItemSetsemSet, globalSupports):

    # Helper function to write the output files in the required format
    freqItemLists = list() # List of all sets at each layer of the transaction tree
    freqItemList = list() # List of all the converted item sets

    for i in range(1, len(globalFreqItemSets) + 1):
        # Convert dicts for each layer into a list of lists
        freqItemLists.append(list(globalFreqItemSets.get(i)))

    for lists in freqItemLists:
        tempItemList = list()

        for items in lists:
            # Convert the dict of frozensets into regular lists
            currentItemList = items
            currentItemList = list(currentItemList)
            tempItemList.append(currentItemList)

        freqItemList.append(tempItemList)

    # Write output file for subtask a
    with open('oneItems.txt', 'w') as ones:
        for supportList, freqItems in zip(globalSupports, freqItemList):
            for support, item in zip(supportList, freqItems):
                ones.write(str(support)+':'+str(item[0])+'\n')
            break
        ones.close()

    # Write output file for subtask b
    with open('patterns.txt', 'w') as patterns:
        for supportList, freqItems in zip(globalSupports, freqItemList):
            for support, items in zip(supportList, freqItems):
                patterns.write(str(support)+':')
                first = True
                for item in items:
                    if first:
                        patterns.write(str(item))
                        first = False
                    else:
                        patterns.write(';'+str(item))
                patterns.write('\n')
        patterns.close()


def aprioriMain(minSup):

    # Main Function to run the apriori algorithm
    globalFreqItemSets = dict() # Dictionary for the frequent item sets at each level of the transaction tree
    globalItemSets = defaultdict(int) # Defaultdict for the Item sets
    globalSupports = list() # List for the absolute supports

    # Generate the first set
    currentSet = aboveMinSup(itemSet, itemSets, minSup, globalItemSets, globalSupports)
    k = 2 # Initialize k as the second level of the transaction tree

    while(currentSet):
        # Iterate through the levels of the tree while the current level returns item sets
        # Set the previous frequent item list to the current set
        globalFreqItemSets[k-1] = currentSet
        # Generate a new candidate set from the union of the current set
        candidateSet = getUnion(currentSet, k)
        # Prune the candidate set
        candidateSet = pruning(candidateSet, currentSet, k-1)
        # Calculate supports for the candidate set
        currentSet = aboveMinSup(candidateSet, itemSets, minSup, globalItemSets, globalSupports)
        
        # Go to the next layer
        k += 1

    # Return the global frequent item set and the global supports
    return globalFreqItemSets, globalSupports

globalFreqItemSets, globalSupports = aprioriMain(0.01)
writeFiles(globalFreqItemSets, globalSupports)
# %%
