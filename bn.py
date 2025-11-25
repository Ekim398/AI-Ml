class Variable:
    def __init__(self, name):
        self.name = name
        self.domain = []        # List of possible values
        self.parents = []       # List of parent Variable objects
        self.children = []      # List of child Variable objects
        self.cpt = {}           # Conditional Probability Table

class BayesianNetwork:
    def __init__(self):
        self.variables = {}     # Dictionary of variables by name

    def add_variable(self, variable):
        self.variables[variable.name] = variable

    def get_variable(self, name):
        return self.variables.get(name)
