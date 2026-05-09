""" note this file is not needed, simply for emily's sanity just to make sure shes not going crazy about the data"""

import math

from utils.preprocessing import AA_ALPHABET
from utils.model import hamming_distance

ALPHABET = AA_ALPHABET

def HammingDistance(seqA: str, seqB: str):
    """
    Calculates number of positions where two sequences differ.
    
    :param seqA: Sequence A
    :type seqA: str
    :param seqB: Sequence B
    :type seqB: str
    """

    if len(seqA) != len(seqB): 
        raise IndexError(f"Hamming Distance: Difference in sequence length: {seqA}, {seqB}")

    return int(round(hamming_distance(seqA, seqB) * len(seqA)))

def w_m(seqs, weights, m, x = 0.3):
    """
    inverse of number of sequences of hamming distance less than xL
    
    :param seqs: list of sequences
    :param m: target index
    :param x: hyperparameter
    """

    L = len(seqs[0])
    length = len(seqs)

    xL = int(x * L)
    target = seqs[m]

    count = 0

    for i in range(length):
        hamming = HammingDistance(seqs[i], target)
        
        if hamming < xL: 
            count += weights[i]

    return 1 / count

def M_eff(seqs, weights):
    """
    effective sequences number = sum_m w_m
    
    :param seqs: list of sequences
    """

    total = 0

    for i in range(len(seqs)):
        total += (w_m(seqs, weights, i) * weights[i])

    return total

def frequency(seqs, weights, a, i):

    M = M_eff(seqs, weights)

    total = 0
    length = len(seqs)

    for idx in range(length):
        if a == seqs[idx][i]:
            total += (w_m(seqs, weights, idx) * weights[idx])
    
    return (1 / M) * total

def cooccurrence(seqs, weights, i, j, a, b):
    
    M = M_eff(seqs, weights)

    total = 0
    length = len(seqs)

    for idx in range(length):
        if a == seqs[idx][i] and b == seqs[idx][j]:
            total += (w_m(seqs, weights, idx) * weights[idx])
    
    return (1 / M) * total

def MI(seqs, weights, i, j) -> float:

    total = 0

    for aminoA in ALPHABET:
        for aminoB in ALPHABET:
            co = cooccurrence(seqs, weights, i, j, aminoA, aminoB)

            if co == 0: continue 

            freqA = frequency(seqs, weights, aminoA, i)
            freqB = frequency(seqs, weights, aminoB, j)

            total += (co * math.log(co / (freqA * freqB)))
    
    return total

