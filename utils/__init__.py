

def build_list(*elem):
    """
    Build a list without None elements
    """

    lst = list()
    for e in elem:
        if e is not None:
            lst.append(e)
