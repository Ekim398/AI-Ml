import sys
import xml.dom.minidom
import itertools
from bn import Variable, BayesianNetwork

def parse_xmlbif(filename):

    doc = xml.dom.minidom.parse(filename)
    network = BayesianNetwork()

    #parse names
    for v in doc.getElementsByTagName("VARIABLE"):
        varname = v.getElementsByTagName("NAME")[0].childNodes[0].nodeValue.strip()
        variable = Variable(varname)
        outcomes = v.getElementsByTagName("OUTCOME")
        variable.domain = [_.childNodes[0].nodeValue.strip() for _ in outcomes] #get text inside each <outcome>
        network.add_variable(variable) #add variable to network

    #Definition & query
    for d in doc.getElementsByTagName("DEFINITION"):
        f = d.getElementsByTagName("FOR")[0].childNodes[0].nodeValue.strip()
        variable = network.get_variable(f)

        #Given = parents
        given_elements = d.getElementsByTagName("GIVEN")
        for parent_elem in given_elements:
            parent_name = parent_elem.childNodes[0].nodeValue.strip()
            parent_var = network.get_variable(parent_name)
            variable.parents.append(parent_var)
            parent_var.children.append(variable)

        probs = get_probabilities(d)

        #CPT table
        if variable.parents:
            parent_domains = [parent.domain for parent in variable.parents]
            parent_value_combinations = list_product(*parent_domains)
        else:
            parent_value_combinations = [()]

        expected_num_probs = len(parent_value_combinations) * len(variable.domain)
        if len(probs) != expected_num_probs:
            print(f"Error: The number of probabilities for variable '{variable.name}' does not match the expected number.")
            print(f"Expected {expected_num_probs} probabilities, but got {len(probs)}.")
            sys.exit(1)

        index = 0
        for parent_values in parent_value_combinations:
            cpt_entry = {}
            for var_value in variable.domain:
                prob = probs[index]
                index += 1
                cpt_entry[var_value] = prob
            variable.cpt[parent_values] = cpt_entry

    return network

def get_probabilities(definition_element):
    #Extracts probabilities from TABLE
    probs = []
    table_elements = definition_element.getElementsByTagName("TABLE")
    if table_elements:
        table_node = table_elements[0]
        # Extract text nodes, ignoring comments and other nodes
        text_content = ''
        for node in table_node.childNodes:
            if node.nodeType == node.TEXT_NODE:
                text_content += node.data.strip() + ' '
        # Now split the text_content into numbers
        probs = [float(x) for x in text_content.strip().split()]
    return probs

def list_product(*args):
    if not args:
        return [()]
    else:
        return list(itertools.product(*args)) #product of conditional probabilities
