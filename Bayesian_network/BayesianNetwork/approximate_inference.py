import random
from exact_inference import topological_sort

def likelihood_weighting(X_name, e, bn, N):
    W = {} #dictionary W for weights
    X = bn.get_variable(X_name)
    for xi in X.domain:
        W[xi] = 0.0
    for a in range(N): #loop N times
        sample, weight = weighted_sample(bn, e) #from below function
        W[sample[X_name]] += weight #add weights to dict for each value of X
    W = normalize(W) #call normalize function for posterior distribution
    return W

def weighted_sample(bn, e): #single weighted sample
    w = 1.0
    x = {} #dict to store sampled values
    vars_list = topological_sort(bn.variables.values())
    for var in vars_list:
        if var.name in e:
            x[var.name] = e[var.name] #if variable in evidence dictionary
            prob = probability(var, x[var.name], x)
            w *= prob #update weight
        # if var not in evidence
        else:
            probs = {}
            for xi in var.domain:
                x_copy = x.copy()
                x_copy[var.name] = xi
                probs[xi] = probability(var, xi, x_copy) #probability given parents
            total = sum(probs.values())
            r = random.uniform(0, total) #random sampling
            cumulative = 0.0
            for xi in var.domain:
                cumulative += probs[xi]
                if r <= cumulative: #select value where r < cumulative
                    x[var.name] = xi #update x with sample value
                    break
    return x, w

def probability(var, value, e):
    if var.parents:
        parent_values = tuple(e[parent.name] for parent in var.parents)
    else:
        parent_values = ()
    return var.cpt[parent_values][value]

def normalize(W): #add all weight and divide by total to get probability
    total = sum(W.values())
    return {k: v / total for k, v in W.items()}
