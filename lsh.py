# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import random
from itertools import combinations

# Making the code deterministic for reproducibility
random.seed(42)

# Global parameters
parameter_file = "default_parameters.ini"  # the main parameters file
# the main path were all the data directories are
data_main_directory = Path("data")
parameters_dictionary = (
    dict()
)  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = (
    dict()
)  # dictionary of the input documents, key = document id, value = the document

# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'


def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == "data":
                parameters_dictionary[key] = config[section][key]
            elif key == "naive":
                parameters_dictionary[key] = bool(config[section][key])
            elif key == "t":
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])

# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'


def read_data(data_path):
    for root, dirs, file in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace("\n", " ")
            file_id = int(file_path.stem)
            document_list[file_id] = doc

# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets


def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))

# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.


def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k

# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix


def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            # print('--- (', i, j, ')')
            # print(docs_Sets[i])
            # print(docs_Sets[j])
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(
                docs_Sets[i], docs_Sets[j]
            )

    return similarity_matrix


# --- METHOD FOR TASK 1 --- #


# Creates the k-Shingles of each document and returns a list of them.

# The k-shingles will be the set of k-words that come in a sentence. After starting on the first word,
# and creating the first set of k words, the method will then go to the next word and create the
# next set of k words. The sets are concatonated and the k-shingle is a string once it is added to the list.


def k_shingles():

    # Holds the k-shingles of each document
    docs_k_shingles = []

    # Finding the value k from the parameters file
    k = parameters_dictionary["k"]

    # Iterating the documents
    for id, document in document_list.items():

        # Removing unwanted spaces on the beginning and end of the document, turning the document to lowercase,
        # and splitting the document on space.
        doc = document.strip().lower().split()

        # If the document is shorter than the k words, then we return the whole document as the shingle list.
        if len(document) < k:
            docs_k_shingles.append([' '.join(document)])
            break

        # Iterating the words in the document, adding the k-shingles together, and concatonating
        # back together with spaces where they were.
        shingle = [' '.join(doc[i:i+k]) for i in range(len(doc)-k+1)]

        # Adding the shingle to the list of shingles for that document.
        docs_k_shingles.append(list(set(shingle)))

    return docs_k_shingles


# --- METHOD FOR TASK 2 --- #


# Creates a signatures set of the documents from the k-shingles list


def signature_set(k_shingles):

    # The lists in docs_sig_sets will contain the same amount of elements as there are unique shingles totally.
    # The lists come in the same order as the documents. We check wether the document has the shingle in it, and then change the
    # value to 1 if it has the value
    docs_sig_sets = []

    # Flattening the list of shingles from the document to one long list of all the shingles,
    # and collect the unique shingles
    shingles = [lists for sublist in k_shingles for lists in sublist]
    unique_shingles = set(shingles)

    # Adding one list for each document, with the same length as the list of unique shingles.
    # This way we can control wether the shingle is present in the document or not
    for i in range(len(k_shingles)):
        docs_sig_sets.append(len(unique_shingles) * [0])

    # We create a list of the unique shingles, so to have a reference to the index of the unique shingles
    list_of_unique_shingles = list(unique_shingles)

    # Creating a dictionary for the indices of the shingles, so it is cost efficient to lookup later
    shingle_index_dict = {}
    i = 0
    for elem in list_of_unique_shingles:
        shingle_index_dict[elem] = i
        i += 1

    # The index of the document in the k_shingles list that we are investigating
    signature_index = 0

    # Iterating through the list of shingles per document
    for doc in k_shingles:

        # Iterating through the unique shingles in the list of shingles
        for shingle in set(doc):

            # Using the dictionary of indices of shingles to set the correct indices to 1
            docs_sig_sets[signature_index][shingle_index_dict[shingle]] = 1

        # Increasing the signature index for the next list of shingles
        signature_index += 1

    return docs_sig_sets


# --- METHOD FOR TASK 3 --- #


# Creates the minHash signatures after simulation of permutations

def help(tuple):
    if tuple[1] == 0:
        return (100, 1)
    return tuple


def minHash(docs_signature_sets):

    # The list where the minHash signatures will be stored
    min_hash_signatures = []

    # Finding the value ps from the parameters file
    ps = parameters_dictionary["permutations"]

    # List to store the permutations
    permutations = []

    # Creating ps number of permutations
    for i in range(ps):

        # Instead of a hash function to randomize the indices, i use random.shuffle()
        indices = list(range(len(docs_signature_sets[0])))
        random.shuffle(indices)
        permutations.append(indices)

    count = 0

    # Iterating through the rows in docs_signature_sets
    for row in docs_signature_sets:
        count += 1
        print(len(docs_signature_sets)-count, 'minHash left')

        # List to store the minHash signatures of the row
        signature = []

        # By use of the method in the for-loop, i am using the permutations to find the minHash signature
        for i in range(ps):
            hash_values = []
            for permutation_value_pair in zip(permutations[i], row):
                p, v = permutation_value_pair
                if v == 1:
                    hash_values.append(p)
            signature.append(min(hash_values))

        # Adding the minHash signature of the row to the minHash signatures list.
        min_hash_signatures.append(signature)

    return min_hash_signatures


# --- METHOD FOR TASK 4 --- #


# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents


def lsh(m_matrix):

    # List of candidate sets of documents for checking similarity
    candidates = []

    # Finding the value r from the parameters file
    r = parameters_dictionary["r"]

    # Calculating the number of bands
    bands = len(m_matrix[0]) / r

    # Finding the value buckets from the parameters file
    bucket = parameters_dictionary["buckets"]

    # Building the empty array for the buckets
    buckets = []
    for i in range(bucket):
        buckets.append([])

    # Iterating through the number of bands
    for b in range(int(bands)):

        # Iterating through the number of rows in that band
        for row_index in range(len(m_matrix)):

            # The vector containing the band we are investigating
            banded_values = m_matrix[row_index][((b + 1) * r) - r: (b + 1) * r]

            # Initializing the number used for the hashing function that determines what bucket the document will go to
            hash_number = 0

            # Creating the number use for the hashing function based on the values in the band
            for banded_values_index in range(len(banded_values)):

                # The code below makes for examle if the band was [7, 1], then the hashing value would become 71
                hash_number += (
                    10 ** (len(banded_values) - banded_values_index - 1)
                ) * banded_values[banded_values_index]

            # The hashing function
            hash_index = hash_number % bucket

            # Inserting the row index, which is the document number, into the corresponding bucket
            buckets[hash_index].append(row_index)

        # Iterating the buckets
        for bck in buckets:

            # The buckets that have more than one element will arrange all combinations possible of the elements, and add the combinations to the possible candidates list.
            if len(bck) > 1:
                candidates.append(list(combinations(bck.copy(), 2)))

            # This method happends for each band, so the bucket is cleared and prepeared for the next band.
            bck.clear()

    # We turn the list of lists to a single list.
    candidates = [pair for sublist in candidates for pair in sublist]

    return candidates


# --- METHOD FOR TASK 5 --- #


# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):

    similarity_matrix = []

    # Initialazing the similarity matrix to be a nxn matrix with the number of docs as n
    for i in range(len(min_hash_matrix)):
        similarity_matrix.append([0]*len(min_hash_matrix))

    # The similarity matrix will have 0 on all elements that are not in the upper triangular matrix,
    # as the elements are just copies of themselves. The diagonal is also not relevant,
    # as it is just the document compared to itself.

    # Finding the value p from the parameters file
    p = parameters_dictionary["permutations"]

    # Iterating trough the unique candidate pairs, and evaluate their similarity.
    for elem in set(candidate_docs):

        # Number of similar elements in the min-hash matrix between the pair.
        sim = 0

        # Making the tuple into a list
        pair = [elem[0], elem[1]]

        # Iterating through the number of permutations and adding up the similar terms.
        for i in range(p):
            if min_hash_matrix[min(pair)][i] == min_hash_matrix[max(pair)][i]:
                sim += 1

        # Adding the similarity divided by the amount of permutations to the similarity matrix to over the diagonal.
        similarity_matrix[min(pair)][max(pair)] = (sim/p)

    return similarity_matrix


# --- METHOD FOR TASK 6 --- #


# Returns the document pairs of over t% similarity


def return_results(lsh_similarity_matrix):

    # Creating the list of document pairs that are over t% similarity
    document_pairs = []

    # Finding the value t from the parameters file
    t = parameters_dictionary["t"]

    # k is the iteration variable
    k = len(lsh_similarity_matrix)

    # We iterate only on the upper triangluar part of the marix, as it's only there
    # there are values that are non-zero
    for i in range(k):
        for j in range(i+1, k):

            # We find the value for the similarity from the lsh_similarity_matrix
            sim = lsh_similarity_matrix[i][j]

            # Check if the similarity is above the t% threshold, and if it is,
            # append it to the list of document pairs
            if sim >= t:
                document_pairs.append((i+1, j+1))

    return document_pairs


# --- METHOD FOR TASK 6 --- #


def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):

    # Initializing the values for the false negative and positive results
    false_negatives = 0
    false_positives = 0

    # Finding the value t from the parameters file
    t = parameters_dictionary["t"]

    # k is the iteration variable
    k = len(lsh_similarity_matrix)

    # We iterate only on the upper triangluar part of the marix, as it's only there
    # there are values that are non-zero
    for i in range(k):
        for j in range(i+1, k):

            # Using the function get_triangle_index(i,j,k) defined in the assignment to find the
            # index of the naive_similarity_matrix that corresponds to the index of our lsh_similarity_matrix
            index = get_triangle_index(i, j, len(lsh_similarity_matrix))

            # Finding the naive similarity
            naive_sim = naive_similarity_matrix[index]

            # Finding the calculated similarity
            sim = lsh_similarity_matrix[i][j]

            # Checking wether the result is a false negative or positive, and storing a value if it is
            if (naive_sim >= t) and (sim <= t):
                false_negatives += 1
            if (naive_sim <= t) and (sim >= t):
                false_positives += 1

    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == "__main__":
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary["data"]
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary["naive"]:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print(
            "Calculating the similarities of",
            len(naive_similarity_matrix),
            "combinations of documents took",
            t3 - t2,
            "sec\n",
        )

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = minHash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(
        candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print(
        "Starting to get the pairs of documents with over ",
        parameters_dictionary["t"],
        "% similarity...",
    )
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("The pairs of documents are:\n")
    for p in pairs:
        print(p)
    print("\n")

    # Count false negatives and positives
    if parameters_dictionary["naive"]:
        print("Starting to calculate the false negatives and positives...")
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(
            lsh_similarity_matrix, naive_similarity_matrix
        )
        t17 = time.time()
        print(
            "False negatives = ",
            false_negatives,
            "\nFalse positives = ",
            false_positives,
            "\n\n",
        )

    if parameters_dictionary["naive"]:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")
