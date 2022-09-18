def list_to_tree(l):
    out = {}
    for i,r in enumerate(l):
        out['{}'.format(i)] = r
    return out