import json
from pathlib import Path
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

class Oligo:

    def __init__(self,
                 huffman_dict_path : Path) -> None:

        with open(huffman_dict_path) as fh:
            self.huff_dict = json.load(fh)

        self.decode_dict =  {y: x for x, y in self.huff_dict.items()}
        self.goldman_table = {
            'A': ['C', 'G', 'T'],
            'C': ['G', 'T', 'A'],
            'G': ['T', 'A', 'C'],
            'T': ['A', 'C', 'G'],
        }

        self.goldman_reverse = {
                'A': {'C': '0', 'G': '1', 'T': '2'},
                'C': {'G': '0', 'T': '1', 'A': '2'},
                'G': {'T': '0', 'A': '1', 'C': '2'},
                'T': {'A': '0', 'C': '1', 'G': '2'}
            }

    def word_to_base3(self,
                      word : str) -> list:
        """
        Transforms a word to the list English-optimized Huffman base3 elements
        """
        return ''.join([self.huff_dict[c] for c in word])
        
    def base3_to_word(self,
                      base3 : list[str]) -> str:
        """
        Transforms a list of Huffman base3 elements to 
        """
        return [self.decode_dict[c] for c in base3]
    
    def ternary_to_goldman(self,
                           digits, start='A'):
        """
        Transform Huffman base3 to codons using Goldman schema
        """
        dna = []
        prev = start
        for d in digits:
            base = self.goldman_table[prev][int(d)]
            dna.append(base)
            prev = base
        return ''.join(dna)
    
    def goldman_to_ternary(self,
                           dna, start='A'):
        '''
        Transform Goldman codons to Huffman base3 
        '''
        tern = []
        prev = start
        for base in dna:
            tern.append(self.goldman_reverse[prev][base])
            prev = base
        return ''.join(tern)

    def huffman_ternary_decode(self,
                               ternary):
        '''
        Transform Huffman base3 to original word
        '''
        rev = {v: k for k, v in self.huff_dict.items()}
        
        decoded = []
        buf = ""
        for d in ternary:
            buf += d
            if buf in rev:     
                decoded.append(rev[buf])
                buf = ""        

        if buf != "":
            raise ValueError("Incomplete ternary string: ended with partial code " + buf)

        return "".join(decoded)


### Quick demo
words = ['DNA', 'storage', 'promising', 'technology', 'future', 'data', 'systems']
records = []

# words -> DNA
for word in words:
    o = Oligo('huffman_dict.json')
    b3 = o.word_to_base3(word)
    seq = o.ternary_to_goldman(b3)
    records.append(SeqRecord(seq=seq,
                             id=word))

with open("def_oligos.fasta", "w") as output_handle:
    SeqIO.write(records, output_handle, "fasta")

### DNA -> words
for seq_record in SeqIO.parse("def_oligos.fasta", "fasta"):
    seq = seq_record.seq
    b3 = o.goldman_to_ternary(seq)
    word = o.huffman_ternary_decode(b3)
    print(word)