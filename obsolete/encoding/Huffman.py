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

'''
Incomplete throw-away Huffman coding sketch; 
Preserved here temporarily
'''

class Huffman:
    class Node:
        def __init__(self,left,right):
            '''
            Build a tree with two trees as children
            '''
            self.left  = left
            self.right = right
            self.pr = left.pr + right.pr
        def __str__(self,nested=0):
            return (" "*nested)+'(%0.2e,\n0:%s,\n1:%s)'%(
                self.pr,
                self.left.__str__(nested+1),
                self.right.__str__(nested+1))
        def walk(self,sequence=''):
            l = self.left
            r = self.right
            symbols = []
            if type(l)==Huffman.Node:
                symbols += l.walk(sequence+'0')
            else:
                symbols += [(l.symbol,sequence+'0')]
            if type(r)==Huffman.Node:
                symbols += r.walk(sequence+'1')
            else:
                symbols += [(r.symbol,sequence+'1')]
            return symbols
    class Leaf:
        def __init__(self,symbol,probability):
            self.symbol = symbol
            self.pr = probability
        def __str__(self,nested=0):
            return " "*nested+"(%s; %0.2e)"%(self.symbol,self.pr)
    def __init__(self,frequencies):
        forest  = [Huffman.Leaf(i,pr) for (i,pr) in enumerate(frequencies)]
        ordered = sorted(forest,key=lambda x:x.pr)
        while len(ordered)>1:
            a = ordered[0]
            b = ordered[1]
            new = Huffman.Node(a,b)
            ordered = [new,]+ordered[2:]
            # this is inefficient
            ordered = sorted(ordered,key=lambda x:x.pr)
        self.root  = ordered[0]
        self.table = dict(self.root.walk())
    def __str__(self):
        return self.root.__str__()
    def encode_packets(self,S):
        return [self.table[s] for s in S]
    def encode(self,S):
        return ''.join(self.encode_packets(S))
    def decode(self,B):
        head = self.root
        decoded = []
        for b in B:
            if b=='0':
                head = head.left
                if type(head) is Huffman.Leaf:
                    decoded.append(head.symbol)
                    head = self.root
            elif b=='1':
                head = head.right
                if type(head) is Huffman.Leaf:
                    decoded.append(head.symbol)
                    head = self.root
        return decoded
