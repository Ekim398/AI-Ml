# exact inference using enumeration by inference
def topological_sort(variables):

    visited = set()
    order = []
    cycles = set()

    #marking algorithm
    def visit(var):
        if var.name in cycles:
            raise Exception("Cyclic") #if var.name in a cycle
        if var.name not in visited:
            cycles.add(var.name)
            for parent in var.parents:
                visit(parent)
            cycles.remove(var.name)
            visited.add(var.name)
            order.append(var)

    for var in variables:
        if var.name not in visited:
            visit(var)
    return order

def enumeration_ask(X_name, e, bn):
    X = bn.get_variable(X_name) #query variable
    Q = {}
    vars_list = topological_sort(bn.variables.values())
    for xi in X.domain: #for each possible value of query
        e_copy = e.copy()
        e_copy[X_name] = xi
        Q[xi] = enumerate_all(vars_list, e_copy)
    Q = normalize(Q) #recursion to add
    return Q


def enumerate_all(vars_list, e):
    if not vars_list:
        return 1.0
    Y = vars_list[0]
    rest_vars = vars_list[1:]
    if Y.name in e: #multiply result by enumeration
        prob = probability(Y, e[Y.name], e)
        return prob * enumerate_all(rest_vars, e)
    else: #if y not in evidence
        total = 0
        for y_value in Y.domain:
            e_copy = e.copy()
            e_copy[Y.name] = y_value
            prob = probability(Y, y_value, e_copy)
            total += prob * enumerate_all(rest_vars, e_copy)
        return total

def probability(var, value, e): #compute probability with the CPT table
    if var.parents:
        parent_values = tuple(e[parent.name] for parent in var.parents)
    else:
        parent_values = ()
    return var.cpt[parent_values][value]


def normalize(Q): #dictionary of probabilities so that they add to 1
    total = sum(Q.values())
    return {k: v / total for k, v in Q.items()}
