def switch_index(l):
    nl = [0]*len(l)
    for i,j in enumerate(l):
        nl[j] = i
    return nl