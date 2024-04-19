class c_code:

    def __init__(self, string="", vars=set()):
        self.code_string = string
        self.vars = vars

    def __repr__(self):
        return self.code_string

    def __str__(self):
        return self.__repr__()
