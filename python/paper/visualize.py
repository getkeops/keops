from graphviz import Digraph
import torch
from torch.autograd import Variable

# dot2tex --figonly -f tikz --autosize -t raw hamiltonian_kernel.dot > hamiltonian_kernel.tex

def make_dot(var, params=None, stored_vars=None, mode = "latex"):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(list(params.values())[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='right',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12", rankdir="BT", ordering="in"))
    seen = set()
    
    if stored_vars is not None :
        stored_vars.reverse()
    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                try :
                    name = stored_vars.pop()
                except IndexError :
                    name = ''
                if mode == "latex" :
                    node_name = '\\tl{%s}{%s}' % (name, size_to_str(var.size()))
                    dot.node(str(id(var)), node_name, fillcolor='blue!10', margin='"0.5"')
                else :
                    node_name = '%s \n %s' % (name, size_to_str(var.size()))
                    dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif hasattr(var, 'variable'):
                u = var.variable
                try :
                    name = param_map[id(u)] if params is not None else ''
                except KeyError :
                    name = ''
                if mode == "latex" :
                    node_name = '\\tl{%s}{%s}' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='red!10', margin='"0.5"')
                else :
                    node_name = '%s \n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor='red')
            else:
                if mode == "latex" :
                    if len(seen) == 0 :
                        last_color = "green!10"
                    else :
                        last_color = "white"
                    dot.node(str(id(var)), "$\partial$"+str(type(var).__name__)[:-8], 
                             fillcolor=last_color, margin='"0.5"')
                else :
                    if len(seen) == 0 :
                        last_color = "green"
                    else :
                        last_color = "white"
                    dot.node(str(id(var)), str(type(var).__name__)[:-8], 
                             fillcolor=last_color, margin='"0.5"')
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(var)), str(id(u[0])))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot
