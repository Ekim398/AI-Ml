import sys
import argparse
from parser import parse_xmlbif
from exact_inference import enumeration_ask
from approximate_inference import likelihood_weighting

def parse_arguments():
    #command line
    parser = argparse.ArgumentParser(description='Bayesian Network Inference')
    parser.add_argument('args', nargs='+', help='Command-line arguments')
    args = parser.parse_args()

    cmd_args = args.args
    if len(cmd_args) < 2:
        parser.print_help()
        sys.exit(1)

    #if first number = int
    try:
        num_samples = int(cmd_args[0])
        mode = 'approximate'
        idx = 1
    except ValueError:
        num_samples = None
        mode = 'exact'
        idx = 0

    filename = cmd_args[idx]
    query_var = cmd_args[idx + 1]
    evidence_list = cmd_args[idx + 2:]

    if len(evidence_list) % 2 != 0:
        print("Error: Variable name and value needed.")
        sys.exit(1)

    evidence = {}
    for i in range(0, len(evidence_list), 2):
        var_name = evidence_list[i]
        value = evidence_list[i + 1]
        evidence[var_name] = value

    return mode, num_samples, filename, query_var, evidence

def main():

    mode, num_samples, filename, query_var, evidence = parse_arguments()
    bn = parse_xmlbif(filename)

    #check query right
    if query_var not in bn.variables:
        print(f"Error: Query variable '{query_var}' not found in the network.")
        sys.exit(1)

    #check evidence vars
    for var_name, value in evidence.items():
        if var_name not in bn.variables:
            print(f"Error: Evidence variable '{var_name}' not found.")
            sys.exit(1)
        if value not in bn.get_variable(var_name).domain:
            print(f"Error: No value '{value}' for variable '{var_name}'.")
            sys.exit(1)

    if mode == 'exact':
        #exact inference
        result = enumeration_ask(query_var, evidence, bn)
        print(f"'{query_var}' Exact Inference:")
        for value in bn.get_variable(query_var).domain:
            print(f"P({query_var}={value} | evidence) = {result.get(value, 0.0)}")
    else:
        #approximate inference
        result = likelihood_weighting(query_var, evidence, bn, num_samples)
        print(f"'{query_var}' & {num_samples} samples Approximate Inference:")
        for value in bn.get_variable(query_var).domain:
            print(f"P({query_var}={value} | evidence) = {result.get(value, 0.0)}")

if __name__ == '__main__':
    main()
