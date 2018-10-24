# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 22:10:03 2018

@author: Hervé
"""

"""
MAXIMIZING XOR
"""

def maximizingXor(l, r):
    r_list = []
    for a in range(l, r+1):
        for b in range(a, r+1):
            result = ''
            a_bin = bin(a)[2:]
            b_bin = bin(b)[2:]
            complement = 0
            if len(a_bin)!=len(b_bin):
                complement = len(b_bin)-len(a_bin)
                result += '1'*complement
            for loop in range(len(a_bin)):
                if a_bin[loop] == b_bin[loop+complement]:
                    result += '0'
                else:
                    result += '1'
            r_list.append(int(result, 2))
    return max(r_list)


"""
SUM vs XOR
"""

def sumXor(n):
    n_bin = bin(n)[2:]
    nb_zero = n_bin.count('0') 
    return 2**nb_zero


"""
FLIPPING BITS
"""

def flippingBits(n):
    nBin = bin(n)[2:]
    newBin = ''
    for i in range(len(nBin)):
        newBin  += '1' if nBin[i] == '0' else '0'
    newBin = newBin.rjust(32, '1')
    return int(newBin,2)


"""
XOR SEQUENCE
"""

def xorSequence(l, r):
    return pattern(l-1)^pattern(r)

def pattern(x):
    if (x%8 == 0 or x%8 ==1):
        return x
    elif (x%8 == 2 or x%8 ==3):
        return 2
    elif (x%8 == 4 or x%8 ==5):
        return x+2
    elif (x%8 == 6 or x%8 ==7):
        return 0
    
# Naive approach

def xorSequenceNAIVE(l, r):
    if l==r:
        return xorList(l)
    nbElement = r-l+1
    if nbElement%2 != 0:
        result = xorList(l)
        for i in range(l+2,r+1,2):
            result = result^i
    else:
        result = l+1
        for i in range(l+3,r+1,2):
            result = result^i
    return result

def xorList(n):
    result = 0
    for i in range(1,n+1):
        result = result^i
    return result


"""
THE GREAT XOR
"""

def theGreatXor(x):
    x_bin = bin(x)[2:]
    length = len(x_bin)
    limit = int('1'*length, 2)
    return limit-x

"""
YET ANOTHER MINIMAX PROBLEM
"""

def anotherMinimaxProblem(a):
    maxInt = max(a)
    maxIntBin = bin(maxInt)[2:]
    maxLength = len(maxIntBin)
    convertBin = lambda x : bin(x)[2:].zfill(maxLength)
    aBin = list(map(convertBin, a))
    while CheckDigit(aBin):
        RemoveDigit(aBin)

    aBin.sort()

    zeroBin, oneBin = Separate(aBin)

    result = []
    for zeroElem in zeroBin:
        for oneElem in oneBin:
            result.append(int(zeroElem,2)^int(oneElem,2))
    
    return min(result)

def CheckDigit(a):
    firstDigit = map(lambda x : x[0], a)
    return (len(set(firstDigit)) == 1)

def RemoveDigit(a):
    return list(map(lambda x : x[1:], a))

def Separate(aSorted):
    zeroBin = []
    oneBin = []
    for binary in aSorted:
        if binary[0]=='0':
            zeroBin.append(binary)
        else:
            oneBin.append(binary)
    return zeroBin, oneBin


"""
SANSA AND XOR
"""

def sansaXor(arr):
    if len(arr)%2 == 0:
        return 0
    else :
        result = arr[0]
        j = 2
        while j < len(arr):
            result = result^arr[j]
            j += 2
        return result
    
    
"""
AND PRODUCT
"""

def andProduct(a, b):
    aBin = bin(a)[2:]
    bBin = bin(b)[2:]
    aLength = len(aBin)
    bLength = len(bBin)

    if aLength != bLength:
        return 0
    
    result = 0
    i = 0
    flag = True
    while (i<aLength and flag):
        flag = (aBin[i]==bBin[i])
        if (flag and aBin[i]=='1'):
            result += 2**(aLength-i-1)
        i+=1
    
    return result


"""
CIPHER
"""

def cipher(n, k, s):
    result = s[-1]
    for i in range(n+k-3,k-2,-1):
        if s[i] == '1':
            if result[-(k-1):].count('1')%2 == 0:
                result += '1'
            else:
                result += '0'
        if s[i] == '0':
            if result[-(k-1):].count('1')%2 == 0:
                result += '0'
            else:
                result += '1'
    return result[::-1]


"""
A OR B
"""

def aOrB(k, a, b, c):
    a = bin(int(a, 16))[2:]
    b = bin(int(b, 16))[2:]
    c = bin(int(c, 16))[2:]
    pad = max(map(len, [a,b,c]))
    a = list('0'*(pad - len(a)) + a)
    b = list('0'*(pad - len(b)) + b)
    c = list('0'*(pad - len(c)) + c)
    
    count = 0
    
    for i in range(pad):
        if c[i] == '0':
            count += int(a[i]) + int(b[i])
            a[i] = '0'
            b[i] = '0'
        else:
            if a[i] == '0' and b[i] == '0':
                count += 1
                b[i] = '1'
        
    if count > k:
        print(-1)
    else:
        i = 0
        k -= count
        while i < pad and k > 0:
            if c[i] == '1':
                if a[i] == '1' and b[i] == '1':
                    k -= 1
                    a[i] = '0'
                elif a[i] == '1' and b[i] == '0' and k >= 2:
                    k -= 2
                    a[i] = '0'
                    b[i] = '1'
            i += 1
        print(hex(int("".join(a), 2))[2:].upper())
        print(hex(int("".join(b), 2))[2:].upper())
        
        
"""
XOR KEY
"""

def xorKeyNAIVE(x, queries):
    result = []
    for query in queries:
        a = int(query[0])
        l = int(query[1])
        r = int(query[2])

        aBin = bin(a)[2:]
        candidate = ''
        for digit in aBin:
            candidate += '1' if digit == '0' else '0'
        if candidate in x[l-1:r]:
            result.append(candidate)
        else:
            maxItem = 0
            for i in range(l-1, r):
                if a^x[i] > maxItem:
                    maxItem = a^x[i]
            result.append(maxItem)
    return result


"""
XOR SUBSEQUENCES
"""

import collections

def xorSubsequence(a):
    xorList = [a[0]]
    xorTemp = a[0]
    for element in a[1:]:
        xorTemp ^= element
        xorList.append(xorTemp)
    #print(xorList)
     
    xorResult = a.copy()
    #print(xorResult)
    for subseq in range(1,len(a)):
        xorResult.append(xorList[subseq])
        for i in range(subseq+1, len(a)):
            xorResult.append(xorList[i]^xorList[i-(subseq+1)])
        #print(xorResult)

    #print(xorResult)
    
    counter = collections.Counter(xorResult)
    print(counter)
    freq = max(val for val in list(counter.values()))
    result = []
    for element in list(counter.keys()):
        if counter[element] == freq:
            result.append(element)
    
    return (min(result), freq)

"""
MAXIMIZING FUNCTION
"""

n, q = map(int, raw_input().strip().split())
def bits():
    w = 0
    yield w
    for v in map(int, raw_input().strip().split()):
        w ^= v
        yield w

c = [[0]*(n+2), [0]*(n+2)]
for i, v in enumerate(bits()):
    for j in xrange(2):
        c[j][i + 1] = c[j][i] + (j == v)

for qq in xrange(q):
    x, y, k = map(int, raw_input().strip().split())
    y += 2
    print (y - x) / 2 * ((y - x + 1) / 2) if k else (c[1][y] - c[1][x]) * (c[0][y] - c[0][x])

