from keopscore.binders.LinkCompile import LinkCompile


class Cpu_link_compile(LinkCompile):

    source_code_extension = "cpp"

    def __init__(self):
        LinkCompile.__init__(self)
        # these are used for command line compiling mode
        self.low_level_code_file = "".encode("utf-8")

        # actual dll to be called
        self.true_dllname = self.gencode_file
        # file to check for existence to detect compilation is needed
        self.file_to_check = self.gencode_file

    def generate_code(self):
        # method to generate the code and compile it
        # generate the code and save it in self.code, by calling get_code method from CpuReduc class :
        self.get_code()
        # write the code in the source file
        self.write_code()
        # retreive some parameters that will be saved into info_file.
        self.tagI = self.red_formula.tagI
        self.dim = self.red_formula.dim
