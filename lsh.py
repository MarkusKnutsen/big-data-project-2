# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import random

# Global parameters
parameter_file = "default_parameters.ini"  # the main parameters file
data_main_directory = Path("data")  # the main path were all the data directories are
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
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(
                docs_Sets[i], docs_Sets[j]
            )

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles():
    docs_k_shingles = []  # holds the k-shingles of each document

    k = parameters_dictionary["k"]

    # Iterating the documents
    for id, document in document_list.items():
        # Allocating a place for the k-shingles list
        docs_k_shingles.append([])

        # Removing unwanted spaces on the beginning and end of the document
        doc = document.strip()

        # If the document is shorter than the k, then we return an empty list for that document
        if len(document) < k:
            break

        for i in range(len(doc) - k + 1):
            shingle = doc[i : i + k]
            docs_k_shingles[id - 1].append(shingle.lower())

    return docs_k_shingles


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    # The lists in docs_sig_sets will contain the same amount of elements as there are shingles totally.
    # The lists come in the same order as the documents. We check wether the document has the shingle in it, and then change the
    # value to 1 if it has the value.
    docs_sig_sets = []

    # Flattening the list of shingles from the document, to one long list of all the shingles.
    shingles = sum(k_shingles, [])

    # Iteration varibale
    iter = -1

    unique_shingles = list(set(shingles))

    # Created the document matrix
    for i in range(len(k_shingles)):
        docs_sig_sets.append(len(unique_shingles) * [0])

    for item in unique_shingles:
        iter += 1

        if iter % 400 == 0:
            print(iter / len(k_shingles), "% completed")

        column_index = 0
        for column in docs_sig_sets:
            if item in k_shingles[column_index]:
                column[iter] = 1
            column_index += 1

    return docs_sig_sets


# Making the code deterministic for reproducibility
random.seed(42)


# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def minHash(docs_signature_sets):
    min_hash_signatures = []

    ps = parameters_dictionary["permutations"]

    # # Need to transpose the docs_signature_sets to visulaize better
    dss_t = list(map(list, zip(*docs_signature_sets)))

    # for elem in dss_t:
    #     print(elem)

    print(100 * "-")

    for row in docs_signature_sets:
        # for row in dss_t:
        signature = []

        permutations = []
        for i in range(ps):
            # Instead of a hash function to randomize the indices, i use random.shuffle()
            indices = list(range(len(row)))
            random.shuffle(indices)
            permutations.append(indices)

        for i in range(ps):
            hash_values = []
            for permutation_value_pair in zip(permutations[i], row):
                p, v = permutation_value_pair
                if v == 1:
                    hash_values.append(p)
            signature.append(min(hash_values))

        min_hash_signatures.append(signature)

    # Need to transpose the min_hash_signatures to visulaize better
    dss_t = list(map(list, zip(*min_hash_signatures)))

    for elem in dss_t:
        print(elem)
    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    candidates = []  # list of candidate sets of documents for checking similarity

    r = parameters_dictionary["r"]
    # Number of bands
    bands = len(m_matrix[0]) / r

    bucket = parameters_dictionary["buckets"]
    buckets = []
    for i in range(bucket):
        buckets.append([])

    # for b in range(int(bands)):
    for b in range(1):
        for row_index in range(len(m_matrix)):
            # Initializing the number used for the hashing function
            hash_number = 0

            # The vector containing the band we arre investigating
            banded_values = m_matrix[row_index][((b + 1) * r) - r : (b + 1) * r]

            # Creating the number use for the hashing function based on the values in the band
            for banded_values_index in range(len(banded_values)):
                hash_number += (
                    10 ** (len(banded_values) - banded_values_index - 1)
                ) * banded_values[banded_values_index]

            # The hashing function
            hash_index = hash_number % bucket

            # Used to visualize in the debugging
            print(
                "Banded values: ",
                banded_values,
                ", Hash number: ",
                hash_number,
                ", Hash index: ",
                hash_index,
                ", Row index: ",
                row_index,
            )

            # Inserting the row index, which is the document number, into the corresponding bucket
            buckets[hash_index].append(row_index)

        # Printing the buckets and their content for visualization
        for elem in buckets:
            print(elem)

        # Lurer på hvordan jeg bruker det til å finne kandidater
        # De lander i forskjellige buckets hver gang, hvorfor er det så tilfeldig? Har det noe å gjøre med min min-hashing?

    return candidates


# # METHOD FOR TASK 5
# # Calculates the similarities of the candidate documents
# def candidates_similarities(candidate_docs, min_hash_matrix):
#     similarity_matrix = []

#     # implement your code here

#     return similarity_matrix


# # METHOD FOR TASK 6
# # Returns the document pairs of over t% similarity
# def return_results(lsh_similarity_matrix):
#     document_pairs = []

#     # implement your code here

#     return document_pairs


# # METHOD FOR TASK 6
# def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
#     false_negatives = 0
#     false_positives = 0

#     # implement your code here

#     return false_negatives, false_positives


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

    # # Naive
    # naive_similarity_matrix = []
    # if parameters_dictionary["naive"]:
    #     print("Starting to calculate the similarities of documents...")
    #     t2 = time.time()
    #     naive_similarity_matrix = naive()
    #     t3 = time.time()
    #     print(
    #         "Calculating the similarities of",
    #         len(naive_similarity_matrix),
    #         "combinations of documents took",
    #         t3 - t2,
    #         "sec\n",
    #     )

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

    # # Candidate similarities
    # print("Starting to calculate similarities of the candidate documents...")
    # t12 = time.time()
    # lsh_similarity_matrix = candidates_similarities(candidate_docs, min_hash_signatures)
    # t13 = time.time()
    # print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # # Return the over t similar pairs
    # print(
    #     "Starting to get the pairs of documents with over ",
    #     parameters_dictionary["t"],
    #     "% similarity...",
    # )
    # t14 = time.time()
    # pairs = return_results(lsh_similarity_matrix)
    # t15 = time.time()
    # print("The pairs of documents are:\n")
    # for p in pairs:
    #     print(p)
    # print("\n")

    # # Count false negatives and positives
    # if parameters_dictionary["naive"]:
    #     print("Starting to calculate the false negatives and positives...")
    #     t16 = time.time()
    #     false_negatives, false_positives = count_false_neg_and_pos(
    #         lsh_similarity_matrix, naive_similarity_matrix
    #     )
    #     t17 = time.time()
    #     print(
    #         "False negatives = ",
    #         false_negatives,
    #         "\nFalse positives = ",
    #         false_positives,
    #         "\n\n",
    #     )

    # if parameters_dictionary["naive"]:
    #     print("Naive similarity calculation took", t3 - t2, "sec")

    # print("LSH process took in total", t13 - t4, "sec")
