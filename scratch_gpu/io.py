def load_ops(path):
    d = json.load(open(path))
    ops = []
    for op in d['ops']:
        if op[0] == 'cz':
            ops.append(('cz', int(op[1]), int(op[2])))
        else:
            U = np.array([complex(r, i) for r, i in op[2]]).reshape(2, 2)
            ops.append(('u', int(op[1]), U))
    return d['N'], d['n'], d['k'], ops


def save_ops(path, N, n, k, czs, init, post, tag, loss):
    ops = []
    for op in slots_to_ops(k, czs, init, post):
        if op[0] == 'cz':
            ops.append(['cz', op[1], op[2]])
        else:
            ops.append(['u', op[1], [[float(z.real), float(z.imag)] for z in np.asarray(op[2]).ravel()]])
    json.dump(dict(N=N, n=n, k=k, ops=ops, tag=tag, ncz=len(czs), loss=loss), open(path, 'w'))


# ------------------------------------------------------------------ main loop
