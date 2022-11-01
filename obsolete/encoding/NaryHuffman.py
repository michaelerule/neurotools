#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import nested_scopes
from __future__ import generators
from __future__ import unicode_literals
from __future__ import print_function
import sys
# more py2/3 compat
from neurotools.system import *

'''
N-ary Huffman encoder
'''

class NaryHuffman:
    class Node:
        def __init__(self,children):
            self.children = children
            self.pr = sum([ch.pr for ch in children])
        def __str__(self,nested=0):
            s = (" "*nested)+'(%0.2e,'%self.pr
            for ch in self.children:
                s += '\n0:%s,'%ch.__str__(nested+1)
            return s+')'
        def walk(self,sequence=()):
            symbols = []
            for i,ch in enumerate(self.children):
                code = sequence+(i,)
                if type(ch)==NaryHuffman.Node:
                    symbols += ch.walk(code)
                else:
                    symbols += [(ch.symbol,code)]
            return symbols
    class Leaf:
        def __init__(self,symbol,probability):
            self.symbol = symbol
            self.pr = probability
        def __str__(self,nested=0):
            return " "*nested+"(%s; %0.2e)"%(self.symbol,self.pr)
    def __init__(self,frequencies,degree=3):
        if degree<2:
            raise ValueError("Degree of a tree must be at least 2");
        forest   = [NaryHuffman.Leaf(i,pr) for (i,pr) in enumerate(frequencies)]
        ordered  = sorted(forest,key=lambda x:x.pr)
        while len(ordered)>=degree:
            ordered  = [NaryHuffman.Node(ordered[:degree]),]+ordered[degree:]
            ordered  = sorted(ordered,key=lambda x:x.pr)
        if len(ordered)>1:
            self.root = NaryHuffman.Node(ordered)
        else:
            self.root = ordered[0]
        self.table = dict(self.root.walk())
    def __str__(self):
        return self.root.__str__()
    def encode_packets(self,S):
        return [self.table[s] for s in S]
    def encode(self,S):
        result = []
        for p in self.encode_packets(S):
            result.extend(p)
        return p
    def decode(self,B):
        head = self.root
        decoded = []
        for b in B:
            head = head.children[b]
            if type(head) is NaryHuffman.Leaf:
                decoded.append(head.symbol)
                head = self.root
        return decoded
