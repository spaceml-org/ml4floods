import os


def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print('{} does not exist, creating'.format(dir))
        os.makedirs(dir)


def cast2(x, ty=int):
    return [ty(_x) for _x in x]


def castdict(d, ty=int):
    for k in d.keys():
        d[k] = cast2(d[k], ty=ty)
    return d


def castmatrix(x, ty=int):
    return [cast2(x1, ty=ty) for x1 in x]


def castlistmatrix(cm):
    return [castmatrix(c) for c in cm.tolist()]
