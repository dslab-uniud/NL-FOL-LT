import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
from concurrent.futures import ProcessPoolExecutor
import itertools
import time
import random
import os
import re
import json
import copy
import pickle
from openai import OpenAI
from openai.lib._parsing._completions import type_to_response_format_param
from dotenv import load_dotenv
from z3 import *




# Global constants
list_bool_operators = ['↔', '→', '∧', '∨', '⊕']


# UNIVERSAL LANGUAGE


# Declaring the constants and the symbol will be used
Constant = DeclareSort('Constant')
# Constants
A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12 = Consts('A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12', Constant)
# variables
x, y, z, u, v, w, a, b, c, d, e, f, l, p, t = Consts('x y z u v w a b c d e f l p t', Constant)
# Predicate symbols
U1 = Function('U1', Constant, BoolSort())
U2 = Function('U2', Constant, BoolSort())
U3 = Function('U3', Constant, BoolSort())
U4 = Function('U4', Constant, BoolSort())
U5 = Function('U5', Constant, BoolSort())
U6 = Function('U6', Constant, BoolSort())
U7 = Function('U7', Constant, BoolSort())
U8 = Function('U8', Constant, BoolSort())
U9 = Function('U9', Constant, BoolSort())

B1 = Function('B1', Constant, Constant, BoolSort())
B2 = Function('B2', Constant, Constant, BoolSort())
B3 = Function('B3', Constant, Constant, BoolSort())
B4 = Function('B4', Constant, Constant, BoolSort())
B5 = Function('B5', Constant, Constant, BoolSort())
B6 = Function('B6', Constant, Constant, BoolSort())
B7 = Function('B7', Constant, Constant, BoolSort())
B8 = Function('B8', Constant, Constant, BoolSort())
B9 = Function('B9', Constant, Constant, BoolSort())

T1 = Function('T1', Constant, Constant, Constant, BoolSort())

arity_universal = {
    'U1': 1, 'U2': 1, 'U3': 1, 'U4': 1, 'U5': 1, 'U6': 1, 'U7': 1, 'U8': 1, 'U9': 1,
    'B1': 2, 'B2': 2, 'B3': 2, 'B4': 2, 'B5': 2, 'B6': 2, 'B7': 2, 'B8': 2, 'B9': 2,
    'T1': 3,
}

dict_const_symbol_object_universal = {
    'A1': A1, 'A2': A2, 'A3':A3, 'A4':A4, 'A5':A5, 'A6':A6, 'A7':A7, 'A8':A8, 'A9':A9, 'A10':A10, 'A11':A11, 'A12':A12,
    'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'w': w,
    'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f, 'l':l, 'p':p, 't':t
}

dict_variable_symbol_object_universal = {
    'x': x, 'y': y, 'z': z, 'u': u, 'v': v, 'w': w, 'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':f, 'l':l, 'p':p, 't':t
}

dict_pred_symbol_object_universal = {
    'U1': U1, 'U2': U2, 'U3': U3, 'U4': U4, 'U5': U5, 'U6': U6, 'U7': U7, 'U8': U8, 'U9': U9,
    'B1': B1, 'B2': B2, 'B3': B3, 'B4': B4, 'B5': B5, 'B6': B6, 'B7': B7, 'B8': B8, 'B9': B9,
    'T1': T1,
}


# Stanford langauge
Constant_Stanford = DeclareSort('Constant_Stanford')


A, B, C, D, E, F = Consts('A B C D E F', Constant_Stanford)
# variables
x, y, z, u, w, c, v = Consts('x y z u w c v', Constant_Stanford)
# Predicate symbols
Cube = Function('Cube', Constant_Stanford, BoolSort())
Tet = Function('Tet', Constant_Stanford, BoolSort())
Dodec = Function('Dodec', Constant_Stanford, BoolSort())
Small = Function('Small', Constant_Stanford, BoolSort())
Medium = Function('Medium', Constant_Stanford, BoolSort())
Large = Function('Large', Constant_Stanford, BoolSort())

Smaller = Function('Smaller', Constant_Stanford, Constant_Stanford, BoolSort())
Larger = Function('Larger', Constant_Stanford, Constant_Stanford, BoolSort())
RightOf = Function('RightOf', Constant_Stanford, Constant_Stanford, BoolSort())
LeftOf = Function('LeftOf', Constant_Stanford, Constant_Stanford, BoolSort())
BackOf = Function('BackOf', Constant_Stanford, Constant_Stanford, BoolSort())
FrontOf = Function('FrontOf', Constant_Stanford, Constant_Stanford, BoolSort())
SameRow = Function('SameRow', Constant_Stanford, Constant_Stanford, BoolSort())
SameCol = Function('SameCol', Constant_Stanford, Constant_Stanford, BoolSort())
SameSize = Function('SameSize', Constant_Stanford, Constant_Stanford, BoolSort())
SameShape = Function('SameShape', Constant_Stanford, Constant_Stanford, BoolSort())
Adjoins = Function('Adjoins', Constant_Stanford, Constant_Stanford, BoolSort())

Between = Function('Between', Constant_Stanford, Constant_Stanford, Constant_Stanford, BoolSort())

arity_Stanford = {
        'Cube': 1, 'Tet': 1, 'Dodec': 1, 'Small': 1, 'Medium': 1, 'Large': 1, 'Smaller': 2, 'Larger': 2, 'RightOf': 2, 'LeftOf':2, 'BackOf':2, 'FrontOf':2, 'SameCol': 2, 'SameRow': 2, 'SameSize': 2, 'SameShape':2, 'Adjoins':2, 'Between':3
}



total_signature = {"Cube": 1, "Tet": 1,"Dodec": 1, "Small": 1,"Smaller": 2,"Large": 1,"Larger": 2,"Medium": 1,"Between": 3,"RightOf": 2,"LeftOf": 2,"BackOf": 2,"FrontOf": 2, "SameRow": 2, "SameCol": 2, "SameSize": 2, "SameShape": 2, "Adjoins": 2, "A":0, "B":0, "C":0, "D":0, "E":0,"F":0}

dict_const_symbol_object_Stanford = {
    'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F,
    'x': x, 'y': y, 'z': z, 'u':u, 'v':v, 'w':w, 'c':c,
}

dict_variable_symbol_object_Stanford = {
    'x': x, 'y': y, 'z': z, 'u':u, 'v':v, 'w':w, 'c':c,
}


dict_pred_symbol_object_Stanford = {
    'Cube': Cube, 'Tet': Tet, 'Dodec': Dodec, 'Small': Small, 'Medium': Medium, 'Large': Large, 'Smaller': Smaller, 'Larger': Larger, 'RightOf': RightOf, 'LeftOf':LeftOf, 'BackOf':BackOf, 'FrontOf':FrontOf, 'SameCol': SameCol, 'SameRow': SameRow, 'SameSize': SameSize, 'SameShape':SameShape, 'Adjoins':Adjoins, 'Between':Between
}





























# _____________
# _____________
# Translation and Modifications ######
# _____________
# _____________


















class FOLFormula:

    def __init__(self, formula: str):
        self.formula = formula.strip()
        self.subformulas = self.parse_formula(self.formula)

    def parse_formula(self, formula):
        """Parses the formula into tokens."""
        # This is a placeholder; actual parsing requires tokenization and tree construction.
        return re.split(r'(\(|\)|\w+|∧|∨|→|¬|↔|∀|∃)', formula)
    
    def get_variables(self):
        """Finds variables in the formula."""
        variables = set()
        tokens = self.formula.split()
        
        for i, token in enumerate(tokens):
            if token in {'∀', '∃'} and i + 1 < len(tokens):
                variables.add(tokens[i + 1])
                
        return variables
    
    def replace_quantifier(self, old_quantifier, new_quantifier, position=0):
        """Replaces a specific occurrence of a quantifier in the formula."""
        if old_quantifier not in ['∀', '∃'] or new_quantifier not in ['∀', '∃']:
            raise ValueError('Not quantifiers in input')
        occurrences = [m.start() for m in re.finditer(re.escape(old_quantifier), self.formula)]
        if position < len(occurrences):
            index = occurrences[position]
            self.formula = self.formula[:index] + new_quantifier + self.formula[index + len(old_quantifier):]
            self.subformulas = self.parse_formula(self.formula)
        else:
            raise ValueError('Position exceeds the number of real occurences')
    
    def replace_operator(self, old_operator, new_operator, position=0):
        """Replaces a specific occurrence of a boolean operator in the formula."""
        if old_operator not in ['∧', '∨', '→', '↔', '⊕'] or new_operator not in ['∧', '∨', '→', '↔', '⊕']:
            raise ValueError('Not operators in input')
        occurrences = [m.start() for m in re.finditer(re.escape(old_operator), self.formula)]
        if position < len(occurrences):
            index = occurrences[position]
            self.formula = self.formula[:index] + new_operator + self.formula[index + len(old_operator):]
            self.subformulas = self.parse_formula(self.formula)
        else:
            raise ValueError('Position exceeds the number of real occurences')   
        
    def toggle_negation(self, atom, position=0):
        """Toggles negation on a specific occurrence of an atomic formula."""
        if atom not in (self.get_signature()).keys():
            raise ValueError('Atom is not correct')
        
        occurrences = [m.start() for m in re.finditer(re.escape(atom), self.formula)]
        if position < len(occurrences):
            index = occurrences[position]
            if self.formula[index-1] not in [' ', '¬', '→', '∨', '↔', '∧', 'x', 'y', 'z', '(']:
                raise ValueError
            if self.formula[max(0, index - 1)] == '¬':
                self.formula = self.formula[:index - 1] + self.formula[index:]
            else:
                self.formula = self.formula[:index] + '¬' + self.formula[index:]
            self.subformulas = self.parse_formula(self.formula)
        else:
            raise ValueError('Position exceeds the number of real occurences')
    
    # def get_signature(self):
    #     """Extracts the signature of the formula (predicate symbols and their arities)."""
    #     predicate_pattern = re.findall(r'([A-Z][a-zA-Z0-9]*)\((.*?)\)', self.formula)
    #     signature = {}
    #     for pred, args in predicate_pattern:
    #         arity = len(args.split(',')) if args.strip() else 0
    #         while pred.startswith(' '):    # remove the spaces before the predicate name
    #             pred = pred[1:]
    #         signature[pred] = arity

    #     constants = re.findall(r'(?<!\w)([A-Z])(?!\w)', self.formula)
    #     for const in constants:
    #         signature[const] = 0
    #     return signature

    def get_signature(self):
        """Extracts the signature of the formula (predicate symbols and their arities)."""
        formula = self.formula
        r,c,v = extract_rel_const_var(formula)

        signature = {}
        for key in r.keys():
            signature[key] = r[key]
        for key in c:
            signature[key] = 0
        return signature 
    
    # def get_signature_new(self):
    #     """Extracts the signature of the formula (predicate symbols and their arities)."""
    #     signature = {}
        
    #     # Identify quantified variables to exclude them
    #     quantified_vars = set()
    #     quant_matches = re.findall(r'[∀∃]\s*([a-zA-Z][a-zA-Z0-9_]*)', self.formula)
    #     quantified_vars.update(quant_matches)
        
    #     # Clean formula by removing quantifier declarations
    #     cleaned_formula = re.sub(r'[∀∃]\s*[a-zA-Z][a-zA-Z0-9_]*\s*', '', self.formula)
        
    #     # Find predicates with proper arity calculation
    #     predicate_pattern = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*)\s*\((.*?)\)', cleaned_formula)
        
    #     for pred, args in predicate_pattern:
    #         pred = pred.strip()
    #         if pred in quantified_vars:
    #             continue
                
    #         if args.strip():
    #             arg_list = [arg.strip() for arg in args.split(',') if arg.strip()]
    #             arity = len(arg_list)
    #         else:
    #             arity = 0
    #         signature[pred] = arity

    #     # Extract constants from predicate arguments
    #     constants = set()
    #     for pred, args in predicate_pattern:
    #         if pred not in quantified_vars and args.strip():
    #             arg_list = [arg.strip() for arg in args.split(',') if arg.strip()]
    #             for arg in arg_list:
    #                 if (re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', arg) and 
    #                     arg not in signature and 
    #                     arg not in quantified_vars):
    #                     constants.add(arg)
        
    #     for const in constants:
    #         signature[const] = 0
            
    #     return signature
    
    def __repr__(self):
        return f"FOLFormula({self.formula})"
    
    def __str__(self):
        return self.formula
    
    def get_quant_modifications(self):
        """ Construct all the modifications related to quantifiers of a given formula"""
        quant_modifications = []
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_quantifier('∀', '∃', position)
                quant_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_quantifier('∃', '∀', position)
                quant_modifications.append(x)
                position += 1
            except ValueError:
                break
        return quant_modifications
        
    def get_boolean_modifications(self):
        boolean_modifications = []
        position = 0
        while True:
            x = copy.deepcopy(self) 
            try:
                x.replace_operator('∧', '∨', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('∧', '→', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('∧', '↔', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break

        position = 0
        while True:
            x = copy.deepcopy(self) 
            try:
                x.replace_operator('⊕', '∧', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self) 
            try:
                x.replace_operator('⊕', '∨', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('⊕', '→', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('⊕', '↔', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        

        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('∨', '∧', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('∨', '→', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('∨', '↔', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('→', '∧', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('→', '∨', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('→', '↔', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('↔', '∧', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('↔', '∨', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        position = 0
        while True:
            x = copy.deepcopy(self)
            try:
                x.replace_operator('↔', '→', position)
                boolean_modifications.append(x)
                position += 1
            except ValueError:
                break
        
        return boolean_modifications

    def get_negation_modifications(self):
        negation_modifications=[]
        signature = self.get_signature()
        for atomic_symbol in signature.keys():
            if signature[atomic_symbol] != 0:
                position = 0
                while True:
                    x = copy.deepcopy(self)
                    try:
                        x.toggle_negation(atomic_symbol, position)
                        negation_modifications.append(x)
                        position += 1
                    except ValueError:
                        break
        return negation_modifications
    
    def replace_atom(self, atom1, atom2, position=0):
        """Replace a predicate symbol with any other
        - atom1 is, for instance, RightOf"""
        signature = self.get_signature()
        if atom1 not in (self.get_signature()).keys():
            raise ValueError('Atom1 is not correct')
        if atom2 not in total_signature.keys():
            raise ValueError('Atom2 is not correct')
        if total_signature[atom1] != total_signature[atom2]:
             raise ValueError("Atoms don't have the same arity")
        occurrences = [m.start() for m in re.finditer(re.escape(atom1), self.formula)]

        for i in range(len(occurrences)):       # Discard the occurences in which atom1 is a substring of atom2
            item = occurrences[i]
            if (self.formula)[item+len(atom1)].isalpha():
                occurrences[i] = -1
        occurrences = [i for i in occurrences if i != -1]

        if position < len(occurrences):
            index = occurrences[position]
            len1 = len(atom1)
            self.formula = self.formula[:index] + atom2 + self.formula[index+len1:]
            self.subformulas = self.parse_formula(self.formula)
        else:
            raise ValueError('Position exceeds the number of real occurences')

    def get_predicate_modifications(self):
        predicate_modifications=[]
        for atom1 in (self.get_signature()).keys():
            for atom2 in total_signature.keys():
                if atom1 != atom2 and total_signature[atom1] == total_signature[atom2] and total_signature[atom1] != 0:
                    position = 0
                    while True:
                        x = copy.deepcopy(self)
                        try:
                            x.replace_atom(atom1, atom2, position)
                            predicate_modifications.append(x)
                            position += 1
                        except ValueError:
                            break 
        return predicate_modifications
    
    def replace_constants(self, const1, const2, position=0):
        """Replace a predicate symbol with any other
        - atom1 is, for instance, RightOf"""
        signature = self.get_signature()
        if const1 not in (self.get_signature()).keys() or total_signature[const1] != 0:
            raise ValueError('Const1 is not correct')
        if const2 not in total_signature.keys() or total_signature[const2] != 0:
            raise ValueError('Const2 is not correct')
        occurrences = [m.start() for m in re.finditer(re.escape(const1), self.formula)]

        for i in range(len(occurrences)):       # Discard the pccurrences of capital letters that are in atoms
            item = occurrences[i]
            if (self.formula)[item+len(const1)].isalpha():
                occurrences[i] = -1
        occurrences = [i for i in occurrences if i != -1]

        if position < len(occurrences):
            index = occurrences[position]
            self.formula = self.formula[:index] + const2 + self.formula[index+1:]
            self.subformulas = self.parse_formula(self.formula)
        else:
            raise ValueError('Position exceeds the number of real occurences')
        
    def get_constant_modifications(self):
        constant_modifications=[]
        for const1 in (self.get_signature()).keys():
            for const2 in total_signature.keys():
                if const1 != const2 and total_signature[const1] == total_signature[const2] and total_signature[const1] == 0:
                    position = 0
                    while True:
                        x = copy.deepcopy(self)
                        try:
                            x.replace_constants(const1, const2, position)
                            constant_modifications.append(x)
                            position += 1
                        except ValueError:
                            break 
        return constant_modifications


    def get_logical_modifications(self):
        logical_modifications = []
        logical_modifications += self.get_quant_modifications()
        logical_modifications += self.get_boolean_modifications()
        logical_modifications += self.get_negation_modifications()
        return logical_modifications

    def get_modifications(self):
        """ Construct all the modifications of a given formula (outcome is the list of modifications) """
        modifications = []
        # quantifier modifications
        modifications += self.get_quant_modifications()

        # boolean operators 
        modifications += self.get_boolean_modifications()

        # negation
        modifications += self.get_negation_modifications()

        # predicate symbols
        modifications += self.get_predicate_modifications()

        # constants
        modifications += self.get_constant_modifications()
        
        return modifications

    def translate(self, translation_rules, meanings, count, negation_meanings={}, type_negation=0, parenthesis=0):
        """
        Translates the FOL formula into natural language while preserving case of constants.
        :return: A natural language translation of the formula.
        count is an index that indicates the combination used for the translation. 
        type_negation can be 0 or 1 and indicates the method in which we translate the negations.
        """
        string_formula = self.formula
    
    # First identify all constants 
        r, constants, v = extract_rel_const_var(string_formula)
        
        
        def helper_implication(result):
            for i in range(len(result)):
                if result[i] != '→':
                    i+=1
                else:
                    break
            end_left = i
            start_right = i
            while result[end_left] != ")":
                end_left -= 1

            j = end_left
            depth = -1
            while depth != 0:
                j -= 1
                if result[j] == '(':
                    depth += 1
                elif result[j] == ')':
                    depth -= 1
            start_left = j
            while result[start_right] != '(':
                start_right += 1
            j = start_right
            depth = 1
            while depth != 0:
                j += 1
                if result[j] == '(':
                    depth += 1
                elif result[j] == ')':
                    depth -= 1
            end_right = j

            left = result[start_left:end_left+1]
            right = result[start_right:end_right+1]
            return start_left, end_right, left, right


        def apply_translation(formula, combo, translation_rules):
            result = formula
            logical_symbols = {'∧', '∨', '→', '↔', '∃', '∀'}
            while {ch for ch in result if ch in logical_symbols} != set():
                for i, rule in enumerate(translation_rules):
                    replacement = combo[i]
                    if rule["type"] == "regex":
                        result = re.sub(rule["pattern"], replacement, result)
                    elif rule["type"] == "string":
                        result = result.replace(rule["pattern"], replacement)
                    elif rule["type"] == "impl":
                        if replacement == "\\1 implies that \\2":
                            result = result.replace('→', "implies that")
                        else:
                            if '→' in result:
                                start, end, left, right = helper_implication(result)
                                sub = f'if {left}, then {right}'
                                result = result[:start] + sub + result[end+1:]
                                
            return result
        
        all_choices = [rule["replacements"] for rule in translation_rules]
        combinations = list(itertools.product(*all_choices))
        translation = apply_translation(string_formula, combinations[count], translation_rules)

    # Handle equality separately
        translation = re.sub(r'([^>]+) = ([^>]+)', r'\1 is equal to \2', translation)
        
    
    # Replace predicates and variables while preserving constants
        signature = self.get_signature()

        for symbol, arity in signature.items():
            if symbol in meanings and symbol not in constants:  # Skip constants
                meaning = meanings[symbol]
                negation_meaning = negation_meanings[symbol]
                if arity == 0:
                    pattern = rf"\b{symbol}\b"
                    translation = re.sub(pattern, meaning, translation)
                else:
                    # Pattern for predicates with arguments
                    args_pattern = r'([^),]+)' + (r'\s*,\s*([^),]+)' * (arity - 1))
                    pattern = rf"{symbol}\s*\(\s*{args_pattern}\s*\)"

                    def replacer(match):
                        args = [g for g in match.groups() if g is not None]
                        # Preserve case of constant arguments
                        formatted_args = []
                        for arg in args:
                            if arg.strip() in constants:
                                formatted_args.append(arg.strip())
                            else:
                                formatted_args.append(arg.lower().strip())
                    
                        return meaning.format(*formatted_args)
        
                    def negation_replacer(match):   
                        """ Do the same thing of the function above replacer, but with this function
                    ¬Cube(A) is translated into 'A isn't a cube.' """
                        
                        args = [g for g in match.groups() if g is not None]
                    # Preserve case of constant arguments
                        formatted_args = []
                        for arg in args:
                            if arg.strip() in constants:
                                formatted_args.append(arg.strip())
                            else:
                                formatted_args.append(arg.lower().strip())
                    
                        return negation_meaning.format(*formatted_args)
                    
                    # Handle the different ways to translate a negation

                    if type_negation == 0:
                        translation = translation.replace("¬", "it's false that ")
                        translation = re.sub(pattern, replacer, translation)
                    elif type_negation == 1:
                        pattern1 = rf"¬{symbol}\s*\(\s*{args_pattern}\s*\)"
                        pattern2 = rf"¬ {symbol}\s*\(\s*{args_pattern}\s*\)"
                        translation = re.sub(pattern1, negation_replacer, translation)
                        translation = re.sub(pattern2, negation_replacer, translation)
                        translation = re.sub(pattern, replacer, translation)
                    else:
                        raise ValueError('type_negation is not a binary value')

    # Translate the negation symbols not relative to atoms
        translation = translation.replace("¬", "it's false that ")
        
    # Clean up spaces
        if parenthesis == 0:
            translation = translation.replace('(', ' ')        
            translation = translation.replace(')', ' ') 
            translation = translation.replace(' ,', ',')
            
        translation = ' '.join(translation.split())
    
    # Final formatting
        translation = translation + '.'
        translation = translation[0].upper() + translation[1:]

        return translation
    
    
def translate_formulas(list_formulas, predicates_meanings, neg_predicates_meanings, translation_rules, combination):
    ''' Given a list of FOLformulas, it returns a list of translated formulas.
    
    - list_formulas can be a list of lists of FOLformulas.
    - translation_rules is a list of dictionaries with the following keys:
        - 'type': 'regex' or 'string' or 'impl'
        - 'pattern': the pattern to be replaced
        - 'replacement': the replacement
    - combination is a list of integers indicating the combination of the translation_rules to be used.
    '''

    assert all(isinstance(i, int) for i in combination), "The combination must be a list of integers"
    
    # fix the combination number according to the convention used
    if len(combination) == 5:
        combination.extend(combination[-2:])
    
    assert len(combination) == 7

    # Compute transl_coefficients so that transl_coefficients[i] is the product of len(translation_rules[j]['replacements']) for j from i+1 to the end (transl_coefficients[-1] = 1)
    def compute_transl_number(translation_rules, combination):
        transl_coefficients = []
        n = len(translation_rules)
        for i in range(n):
            prod = 1
            for j in range(i+1, n):
                prod *= len(translation_rules[j]['replacements'])
            transl_coefficients.append(prod)
        
        number_translation = 0
        for i in range(len(combination)):
            number_translation += combination[i]*transl_coefficients[i]
        return number_translation

    def translate_formulas_single_list(list_formulas, predicates_meanings, neg_predicates_meanings, translation_rules, combination):
        list_translated_formulas = []
        for i, formula in enumerate(list_formulas):
            if not isinstance(formula, FOLFormula):
                list_formulas[i] = FOLFormula(formula)
            number_translation = compute_transl_number(translation_rules, combination[:-2])
            type_negation = combination[-2]
            parenthesis = combination[-1]
            translated_formula = formula.translate(translation_rules, predicates_meanings, number_translation, negation_meanings = neg_predicates_meanings, type_negation = type_negation, parenthesis = parenthesis)
            list_translated_formulas.append(translated_formula)

        return list_translated_formulas
    
    def translate_formulas_list_list(list_formulas, predicates_meanings, neg_predicates_meanings, translation_rules, combination):
        list_translated_formulas = [[] for i in range(len(list_formulas))]
        for i, item in enumerate(list_formulas):
            translated_item = translate_formulas_single_list(item, predicates_meanings, neg_predicates_meanings, translation_rules, combination)
            list_translated_formulas[i] = translated_item

        return list_translated_formulas
    
    if isinstance(list_formulas[0], list):
        return translate_formulas_list_list(list_formulas, predicates_meanings, neg_predicates_meanings, translation_rules, combination)
    else:
        return translate_formulas_single_list(list_formulas, predicates_meanings, neg_predicates_meanings, translation_rules, combination)


def translate_formulas_FOLIO(fol_formula_or_list, constant_meaning, relational_meaning_pos, relational_meaning_neg, translation_rules, combination):
    '''fol_formulas_or_list can be a single or a list of FOLFormulas'''

    def in_the_right_format(fol_formula):
        assert isinstance(fol_formula, FOLFormula)
        x, dict = put_universal(fol_formula.formula)
        x = parse_formula_SMTLIB_universal(x)
        x = z3_to_logical_notation(x)
        x = re.sub("|".join(dict), lambda x: dict[x.group(0)], x)
        x = FOLFormula(x)
        return x
    
    def translate_formulas_FOLIO_single(fol_formula):
        x = in_the_right_format(fol_formula)
        y = translate_formulas([x], relational_meaning_pos, relational_meaning_neg, translation_rules, combination)[0]
        y = y[0].lower() + y[1:]

        if len(constant_meaning) != 0:
            y = re.sub("|".join(constant_meaning), lambda x: constant_meaning[x.group(0)], y)

        y = y[0].upper() + y[1:]
        return y
        
    if isinstance(fol_formula_or_list, list):
        if isinstance(fol_formula_or_list[0], list):
            assert isinstance(fol_formula_or_list[0][0], FOLFormula)
            return [[translate_formulas_FOLIO_single(s) for s in z] for z in fol_formula_or_list]
        else:
            return [translate_formulas_FOLIO_single(s) for s in fol_formula_or_list]
    else:
        return translate_formulas_FOLIO_single(fol_formula_or_list)
    




def modify_formulas(modification_type, list_formulas, k = None, seed: int = None):
    ''' Given a list of formulas, it returns a list of list of modified formulas.

    modification_type is a string that can be ('quantifier', 'boolean', 'negation', 'predicate', 'constant', 'all', 'all_logical')

    k is the number of modifications to be generated for each formula. If None, all the modifications are generated.

    The output is a the list of list of modified formulas
    
    '''

    if modification_type == 'boolean':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_boolean_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_boolean_modifications() for formula in list_formulas]
    elif modification_type == 'quantifier':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_quant_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_quant_modifications() for formula in list_formulas]
    elif modification_type == 'negation':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_negation_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_negation_modifications() for formula in list_formulas]
    elif modification_type == 'predicate':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_predicate_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_predicate_modifications() for formula in list_formulas]
    elif modification_type == 'constant':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_constant_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_constant_modifications() for formula in list_formulas]
    elif modification_type == 'all':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_modifications() for formula in list_formulas]
    elif modification_type == 'all_logical':
        if k is not None:
            return [shuffle_lists_take_k(formula.get_logical_modifications(), k) for formula in list_formulas]
        else:
            return [formula.get_logical_modifications() for formula in list_formulas]
    else:
        raise ValueError('Modification type not supported.')
    
def get_negations(list_formulas):
    ''' Get the negation of the list of formulas'''
    assert isinstance(list_formulas, list)
    list_negations = []
    for formula in list_formulas:
        neg1 = FOLFormula('¬(' + formula.formula + ')')

        neg2, dict_replacements = put_universal(formula.formula)
        neg2 = parse_formula_SMTLIB_universal(neg2)
        neg2 = negate_to_nnf(neg2)
        neg2 = z3_to_logical_notation(neg2)
        for key, item in dict_replacements.items():
            neg2 = neg2.replace(key, item)
        neg2 = neg2.replace('  ','')

        list_negations.append([neg1, FOLFormula(neg2)])
    return list_negations

def compute_if_free_variable(formula, subformula):
    r,c,v = extract_rel_const_var(formula)
    for x in v:
        if x in subformula:
            if (f'∃{x}' in subformula) or (f'∀{x}' in subformula):
                free_variables = False
            else:
                return True
    return False

# Use case for Stanford



def parse_formula_Stanford(string):
    '''Put the formula string in the SMTLIB format'''
    # remove spaces and initial parenthesis
    string = string.replace(' ', '')

    def parenthesis_helper(s, position_parenthesis):
        assert s[position_parenthesis] == '('
        t = s[position_parenthesis:]
        count = 1
        position = 1
        while count != 0:
            try:
                if t[position] == '(':
                    count += 1
                elif t[position] == ')':
                    count -= 1
                position += 1
            except:
                raise IndexError('Parenthesis are not correct')
        return position_parenthesis + position -1
    

    def split_top_level(s):
        """Split the formula at the top-level boolean operator, respecting parentheses."""
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c in list_bool_operators:
                return s[:i], c, s[i+1:]
        return None
    
    def split_top_level_given_op(s, op):
        '''Split the formula at the top-level boolean operator given.
        Take also in consideration that ∨, ∧ and ⊕ associates to the left, 
        → and ↔ to the right '''
        if op == '∧' or op == '⊕' or op == '∨':
            list = []
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif depth == 0 and c == op:
                    list.append((s[:i], c, s[i+1:]))
            if len(list) == 0:
                return None
            else:
                return list[-1]
        else:
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif depth == 0 and c == op:
                    return s[:i], c, s[i+1:]
            return None


    # Delete the useless parenthesis at the beginning of the formula
    condition = True
    while condition:
        condition = ((string[0] == '(') and (parenthesis_helper(string, 0) == len(string)-1))
        if condition:
            string = string[1:-1]
        else:
            break
    
    # Deal with operator precedences
    if split_top_level_given_op(string, '↔'):
        left, op, right = split_top_level_given_op(string, '↔')
        return And(Implies(parse_formula_Stanford(left), parse_formula_Stanford(right)), Implies(parse_formula_Stanford(right), parse_formula_Stanford(left)))
    elif split_top_level_given_op(string, '→'):
        left, op, right = split_top_level_given_op(string, '→')
        return Implies(parse_formula_Stanford(left), parse_formula_Stanford(right))
    elif split_top_level_given_op(string, '∨'):
        left, op, right = split_top_level_given_op(string, '∨')
        return Or(parse_formula_Stanford(left), parse_formula_Stanford(right))
    elif split_top_level_given_op(string, '∧'):
        left, op, right = split_top_level_given_op(string, '∧')
        return And(parse_formula_Stanford(left), parse_formula_Stanford(right))
    elif split_top_level_given_op(string, '⊕'):
        left, op, right = split_top_level_given_op(string, '⊕')
        return Or(And(Not(parse_formula_Stanford(left)), parse_formula_Stanford(right)), And(parse_formula_Stanford(left), Not(parse_formula_Stanford(right))))
    
    # universal case: 
    if string[0] == '∀':
        list_variables = []
        for variable in dict_variable_symbol_object_Stanford.keys():
            if string.find(variable) == 1:
                list_variables.append(variable)
        var1 = max(list_variables, key = lambda s: len(s))
        # case ∀x(...) or ∀x∀y(...)
        if ((string[1+len(var1)] == '(') or  (string[1+len(var1)] == '∀') or  (string[1+len(var1)] == '∃')):
            return ForAll(dict_variable_symbol_object_Stanford[var1], parse_formula_Stanford(string[1+len(var1):]))
        # case ∀x... withou boolean operator (they have the precedence)
        elif split_top_level(string) is None:
            return ForAll(dict_variable_symbol_object_Stanford[var1], parse_formula_Stanford(string[1+len(var1):]))
        
    # existential case:
    if string[0] == '∃':
        list_variables = []
        for variable in dict_variable_symbol_object_Stanford.keys():
            if string.find(variable) == 1:
                list_variables.append(variable)
        var1 = max(list_variables, key = lambda s: len(s))
        # case ∃x(...) or ∃x∀y(...)
        if ((string[1+len(var1)] == '(') or  (string[1+len(var1)] == '∀') or  (string[1+len(var1)] == '∃')):
            return Exists(dict_variable_symbol_object_Stanford[var1], parse_formula_Stanford(string[1+len(var1):]))
        # case ∃x... without any boolean operator
        elif split_top_level(string) is None:
            return Exists(dict_variable_symbol_object_Stanford[var1], parse_formula_Stanford(string[1+len(var1):]))

    # case of a formula of the form '¬...' 
    if string[0] == '¬':
        # case of a formula of the form '¬(...)'
        if string[1] == '(' and parenthesis_helper(string, 1) == len(string)-1:
            return Not(parse_formula_Stanford(string[1:]))
        # case '¬...'; we treat only the case of a litteral
        elif split_top_level(string) is None:
            return Not(parse_formula_Stanford(string[1:]))

    else:
        for predicate in dict_pred_symbol_object_Stanford.keys():
            if ((string.startswith(predicate)) and (string[len(predicate)] == '(')):
                assert (parenthesis_helper(string, len(predicate)) == len(string)-1)
                # Dealing with constants or variables longer than one charachter
                if arity_Stanford[predicate] == 1:
                    var1 = string[len(predicate)+1:-1]
                    assert ',' not in var1 
                    assert var1 in dict_const_symbol_object_Stanford.keys()
                    return dict_pred_symbol_object_Stanford[predicate](dict_const_symbol_object_Stanford[var1])
                elif arity_Stanford[predicate] == 2:
                    variables = string[len(predicate)+1:-1]
                    assert len(variables.split(',')) == 2
                    var1 = variables.split(',')[0]
                    var2 = variables.split(',')[1]
                    assert var1 in dict_const_symbol_object_Stanford.keys()
                    assert var2 in dict_const_symbol_object_Stanford.keys()
                    return dict_pred_symbol_object_Stanford[predicate](dict_const_symbol_object_Stanford[var1], dict_const_symbol_object_Stanford[var2])
                elif arity_Stanford[predicate] == 3:
                    variables = string[len(predicate)+1:-1]
                    assert len(variables.split(',')) == 3
                    var1 = variables.split(',')[0]
                    var2 = variables.split(',')[1]
                    var3 = variables.split(',')[2]
                    assert var1 in dict_const_symbol_object_Stanford.keys()
                    assert var2 in dict_const_symbol_object_Stanford.keys()
                    assert var3 in dict_const_symbol_object_Stanford.keys()
                    return dict_pred_symbol_object_Stanford[predicate](dict_const_symbol_object_Stanford[var1], dict_const_symbol_object_Stanford[var2], dict_const_symbol_object_Stanford[var3])

def check_equivalence_Stanford(formula1 : str, formula2 : str):
    s = Solver()
    try:
        f1 = parse_formula_Stanford(formula1)
    except:
        return 'Parsing_error_1'
    try:
        f2 = parse_formula_Stanford(formula2)
    except:
        return 'Parsing_error_2'
    s.add(Not(f1 == f2))
    # Add some axioms for the configurations we have
    s.add(ForAll([x,y], Smaller(x,y) == Larger(y,x)))
    s.add(ForAll([x,y], RightOf(x,y) == LeftOf(y,x)))
    s.add(ForAll([x,y], BackOf(x,y) == FrontOf(y,x)))


    s.add(ForAll([x,y], SameRow(x,y) == SameRow(y,x)))
    s.add(ForAll([x,y], SameCol(x,y) == SameCol(y,x)))
    s.add(ForAll([x,y], SameSize(x,y) == SameSize(y,x)))
    s.add(ForAll([x,y], SameShape(x,y) == SameShape(y,x)))
    s.add(ForAll([x,y], Adjoins(x,y) == Adjoins(y,x)))

    s.add(ForAll([x,y,z], Between(x,y,z) == Between(x,z,y)))
    try:
        res = s.check()
        if res == unsat:
            return True
        elif res == sat:
            return False
        else:
            return 'Solver_failed'
    except:
        return 'Solver_failed'



def parse_formula_SMTLIB_universal(string):
    '''Put the formula string in the SMTLIB format, for Z3'''
    # remove spaces and initial parenthesis
    string = string.replace(' ', '')

    def parenthesis_helper(s, position_parenthesis):
        assert s[position_parenthesis] == '('
        t = s[position_parenthesis:]
        count = 1
        position = 1
        while count != 0:
            try:
                if t[position] == '(':
                    count += 1
                elif t[position] == ')':
                    count -= 1
                position += 1
            except:
                raise IndexError('Parenthesis are not correct')
        return position_parenthesis + position -1
    
    def split_top_level(s):
        """Split the formula at the top-level boolean operator, respecting parentheses."""
        depth = 0
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c in list_bool_operators:
                return s[:i], c, s[i+1:]
        return None
    
    def split_top_level_given_op(s, op):
        '''Split the formula at the top-level boolean operator given.
        Take also in consideration that ∨, ∧ and ⊕ associates to the left, 
        → and ↔ to the right '''
        if op == '∧' or op == '⊕' or op == '∨':
            list = []
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif depth == 0 and c == op:
                    list.append((s[:i], c, s[i+1:]))
            if len(list) == 0:
                return None
            else:
                return list[-1]
        else:
            depth = 0
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                elif depth == 0 and c == op:
                    return s[:i], c, s[i+1:]
            return None




    # Delete the useless parenthesis at the beginning of the formula
    condition = True
    while condition:
        condition = ((string[0] == '(') and (parenthesis_helper(string, 0) == len(string)-1))
        if condition:
            string = string[1:-1]
        else:
            break
    
    # Deal with operator precedences
    if split_top_level_given_op(string, '↔'):
        left, op, right = split_top_level_given_op(string, '↔')
        
        return And(Implies(parse_formula_SMTLIB_universal(left), parse_formula_SMTLIB_universal(right)), Implies(parse_formula_SMTLIB_universal(right), parse_formula_SMTLIB_universal(left)))
    elif split_top_level_given_op(string, '→'):
        left, op, right = split_top_level_given_op(string, '→')
        
        return Implies(parse_formula_SMTLIB_universal(left), parse_formula_SMTLIB_universal(right))
    elif split_top_level_given_op(string, '∨'):
        left, op, right = split_top_level_given_op(string, '∨')
        
        return Or(parse_formula_SMTLIB_universal(left), parse_formula_SMTLIB_universal(right))
    elif split_top_level_given_op(string, '∧'):
        
        left, op, right = split_top_level_given_op(string, '∧')
        return And(parse_formula_SMTLIB_universal(left), parse_formula_SMTLIB_universal(right))
    elif split_top_level_given_op(string, '⊕'):
        left, op, right = split_top_level_given_op(string, '⊕')
        
        return Or(And(Not(parse_formula_SMTLIB_universal(left)), parse_formula_SMTLIB_universal(right)), And(parse_formula_SMTLIB_universal(left), Not(parse_formula_SMTLIB_universal(right))))
    
    # universal case: 
    elif string[0] == '∀':
        list_variables = []
        for variable in dict_variable_symbol_object_universal.keys():
            if string.find(variable) == 1:
                list_variables.append(variable)
        var1 = max(list_variables, key = lambda s: len(s))
        # case ∀x(...) or ∀x∀y(...)
        if ((string[1+len(var1)] == '(') or  (string[1+len(var1)] == '∀') or  (string[1+len(var1)] == '∃')):
            return ForAll(dict_variable_symbol_object_universal[var1], parse_formula_SMTLIB_universal(string[1+len(var1):]))
        # case ∀x... withou boolean operator (they have the precedence)
        elif split_top_level(string) is None:
            return ForAll(dict_variable_symbol_object_universal[var1], parse_formula_SMTLIB_universal(string[1+len(var1):]))
        
    # existential case:
    elif string[0] == '∃':
        list_variables = []
        for variable in dict_variable_symbol_object_universal.keys():
            if string.find(variable) == 1:
                list_variables.append(variable)
        var1 = max(list_variables, key = lambda s: len(s))
        # case ∃x(...) or ∃x∀y(...)
        if ((string[1+len(var1)] == '(') or  (string[1+len(var1)] == '∀') or  (string[1+len(var1)] == '∃')):
            return Exists(dict_variable_symbol_object_universal[var1], parse_formula_SMTLIB_universal(string[1+len(var1):]))
        # case ∃x... without any boolean operator
        elif split_top_level(string) is None:
            return Exists(dict_variable_symbol_object_universal[var1], parse_formula_SMTLIB_universal(string[1+len(var1):]))

    # case of a formula of the form '¬...' 
    elif string[0] == '¬':
        # case of a formula of the form '¬(...)'
        if string[1] == '(' and parenthesis_helper(string, 1) == len(string)-1:
            return Not(parse_formula_SMTLIB_universal(string[1:]))
        # case '¬...'; we treat only the case of a litteral
        elif split_top_level(string) is None:
            return Not(parse_formula_SMTLIB_universal(string[1:]))

    else:
        for predicate in dict_pred_symbol_object_universal.keys():
            if ((string.startswith(predicate)) and (string[len(predicate)] == '(')):
                assert (parenthesis_helper(string, len(predicate)) == len(string)-1)
                # Dealing with constants or variables longer than one charachter
                if arity_universal[predicate] == 1:
                    var1 = string[len(predicate)+1:-1]
                    assert ',' not in var1 
                    assert var1 in dict_const_symbol_object_universal.keys()
                    return dict_pred_symbol_object_universal[predicate](dict_const_symbol_object_universal[var1])
                elif arity_universal[predicate] == 2:
                    variables = string[len(predicate)+1:-1]
                    assert len(variables.split(',')) == 2
                    var1 = variables.split(',')[0]
                    var2 = variables.split(',')[1]
                    assert var1 in dict_const_symbol_object_universal.keys()
                    assert var2 in dict_const_symbol_object_universal.keys()
                    return dict_pred_symbol_object_universal[predicate](dict_const_symbol_object_universal[var1], dict_const_symbol_object_universal[var2])
                elif arity_universal[predicate] == 3:
                    variables = string[len(predicate)+1:-1]
                    assert len(variables.split(',')) == 3
                    var1 = variables.split(',')[0]
                    var2 = variables.split(',')[1]
                    var3 = variables.split(',')[2]
                    assert var1 in dict_const_symbol_object_universal.keys()
                    assert var2 in dict_const_symbol_object_universal.keys()
                    assert var3 in dict_const_symbol_object_universal.keys()
                    return dict_pred_symbol_object_universal[predicate](dict_const_symbol_object_universal[var1], dict_const_symbol_object_universal[var2], dict_const_symbol_object_universal[var3])



def check_equivalence_with_parsed_formulas(formula1, formula2):
    s = Solver()
    s.add(Not(formula1 == formula2))
    try:
        res = s.check()
        if res == unsat:
            return True
        elif res == sat:
            return False
        else:
            return 'Solver_failed'
    except:
        return 'Solver_failed'































def extract_rel_const_var(formula: str):
    """
    Parse a First-Order Logic formula using a robust 3-step approach:
    1. Extract variables from quantifiers
    2. Extract constants from predicate arguments
    3. Extract predicates as remaining non-logical symbols
    
    Args:
        formula: String representation of the FOL formula
        
    Returns:
        Tuple (relations, constants, variables) where:
        - relations: Dict mapping relation names to their arity
        - constants: Set of constant symbols
        - variables: Set of variable symbols
    """
    
    # Define logical symbols to exclude
    logical_symbols = {'¬', '→', '↔', '⊕', '∧', '∨', '∃', '∀', '(', ')', ','}
    
    # Step 1: Find all quantified variables (before removing spaces)
    variables = set()
    quantifier_pattern = r'[∀∃]([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(quantifier_pattern, formula):
        variables.add(match.group(1))
    
    # Remove spaces more carefully to avoid affecting quantifiers
    # Strategy: Only remove spaces between probable predicate names and parentheses
    # A predicate name is likely if it's not a single variable after a quantifier
    
    # First, let's identify quantified variables to avoid them
    temp_variables = set()
    for match in re.finditer(r'[∀∃]\s*([a-zA-Z_][a-zA-Z0-9_]*)', formula):
        temp_variables.add(match.group(1))
    
    # Now remove spaces between non-variable identifiers and parentheses
    formula_fixed = formula
    # Find all cases of identifier followed by space and parenthesis
    for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+\(', formula):
        identifier = match.group(1)
        # Only remove space if this identifier is not a quantified variable
        if identifier not in temp_variables:
            formula_fixed = formula_fixed.replace(match.group(0), identifier + '(', 1)
    
    # Step 2: Extract constants from predicate arguments
    constants = set()
    
    # Find all content within parentheses (predicate arguments)
    paren_content_pattern = r'\(([^())]+)\)'
    
    for match in re.finditer(paren_content_pattern, formula_fixed):
        args_str = match.group(1)
        
        # Split arguments by comma, but be careful with nested structures
        args = []
        paren_count = 0
        current_arg = ""
        
        for char in args_str:
            if char == ',' and paren_count == 0:
                if current_arg.strip():
                    args.append(current_arg.strip())
                current_arg = ""
            else:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                current_arg += char
        
        # Add the last argument
        if current_arg.strip():
            args.append(current_arg.strip())
        
        # Process each argument
        for arg in args:
            # Clean the argument
            cleaned_arg = arg.strip()
            
            # Skip empty arguments
            if not cleaned_arg:
                continue
            
            # If it's not a known variable, it's a constant
            if cleaned_arg not in variables:
                # Additional validation: constants shouldn't contain logical symbols
                if not any(sym in cleaned_arg for sym in logical_symbols):
                    constants.add(cleaned_arg)

    # Remove the varibales that are considered constants:
    for c in constants:
        if c in variables:
            constants.remove(c)

    # Step 3: Extract predicate symbols
    relations = {}
    
    # Find all predicate applications: predicate_name(arguments)
    # We'll match any non-logical symbol sequence followed by parentheses
    predicate_pattern = r'([^¬→↔⊕∧∨∃∀(),\s]+)\(([^)]*)\)'
    
    for match in re.finditer(predicate_pattern, formula_fixed):
        predicate_name = match.group(1).strip()
        args_str = match.group(2)
        
        # Skip if the predicate name contains only logical symbols
        if all(c in logical_symbols or c.isspace() for c in predicate_name):
            continue
        
        # Count arguments
        if not args_str.strip():
            arity = 0
        else:
            # Split arguments by comma, handling nested parentheses
            args = []
            paren_count = 0
            current_arg = ""
            
            for char in args_str:
                if char == ',' and paren_count == 0:
                    if current_arg.strip():
                        args.append(current_arg.strip())
                    current_arg = ""
                else:
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                    current_arg += char
            
            # Add the last argument
            if current_arg.strip():
                args.append(current_arg.strip())
            
            arity = len(args)
        
        # Record the predicate and its arity
        if predicate_name in relations:
            # Verify consistent arity
            if relations[predicate_name] != arity:
                raise ValueError(f"Inconsistent arity for predicate {predicate_name}: {relations[predicate_name]} vs {arity}")
        else:
            relations[predicate_name] = arity
    
    return relations, constants, variables



def put_universal(formula: str):
    ''' Put a formula (string in the standard FOL syntax) into the universal language with U1, U2, ..., B1, B2, ...
    It returns the string of the formula in the universal language and the dictionary that represent the replacements done U1: Pred1, ...'''
    assert isinstance(formula, str)

    r,c,v = extract_rel_const_var(formula)
    list_constants = sorted(list(c), key=len, reverse=True)
    unary_symbols = [rel for rel in r.keys() if r[rel] == 1]
    binary_symbols = [rel for rel in r.keys() if r[rel] == 2]
    ternary_symbols = [rel for rel in r.keys() if r[rel] == 3]
    
    list_replacements = sorted(list_constants + unary_symbols + binary_symbols + ternary_symbols, key=len, reverse=True)

    dict_replacements = {}
    x = formula
    position_already_changed = []

    for item in list_replacements:
        c = 0

        if item in list_constants:
            index = list_constants.index(item)+1
            replacement = 'A'+str(index)
            positions = [match.start() for match in re.finditer(item, x)]

            for position in positions:
                position += c*(len(replacement)-len(item))
                if position not in position_already_changed:
                    x = x[:position] + replacement + x[position+len(item):]
                    # update the changed position
                    position_already_changed_new = []
                    for pos in position_already_changed:
                        if pos < position:
                            position_already_changed_new.append(pos)
                        if pos > position:
                            position_already_changed_new.append(pos+len(replacement)-len(item))
                    for i in range(position, position + len(replacement)):
                        position_already_changed_new.append(i) 
                    # update the position in positions with the c coefficient
                    c += 1
                    position_already_changed = position_already_changed_new
            dict_replacements['A' + str(index)] = item

        if item in unary_symbols:
            index = unary_symbols.index(item)+1
            replacement = 'U'+str(index)
            positions = [match.start() for match in re.finditer(item, x)]
            
            for position in positions:
                position += c*(len(replacement)-len(item))
                if position not in position_already_changed:
                    x = x[:position] + replacement + x[position+len(item):]
                    # update the changed position
                    position_already_changed_new = []
                    for pos in position_already_changed:
                        if pos < position:
                            position_already_changed_new.append(pos)
                        if pos > position:
                            position_already_changed_new.append(pos+len(replacement)-len(item))
                    for i in range(position, position + len(replacement)):
                        position_already_changed_new.append(i) 
                    # update the position in positions
                    c += 1
                    position_already_changed = position_already_changed_new
            dict_replacements['U' + str(index)] = item

        if item in binary_symbols:
            index = binary_symbols.index(item)+1
            replacement = 'B'+str(index)
            positions = [match.start() for match in re.finditer(item, x)]
            for position in positions:
                position += c*(len(replacement)-len(item))
                if position not in position_already_changed:
                    x = x[:position] + replacement + x[position+len(item):]
                    # update the changed position
                    position_already_changed_new = []
                    for pos in position_already_changed:
                        if pos < position:
                            position_already_changed_new.append(pos)
                        if pos > position:
                            position_already_changed_new.append(pos+len(replacement)-len(item))
                    for i in range(position, position + len(replacement)):
                        position_already_changed_new.append(i) 
                    # update the position in positions
                    c += 1
                    position_already_changed = position_already_changed_new
            dict_replacements['B' + str(index)] = item

        if item in ternary_symbols:
            index = ternary_symbols.index(item)+1
            replacement = 'T'+str(index)
            positions = [match.start() for match in re.finditer(item, x)]
            for position in positions:
                position += c*(len(replacement)-len(item))
                if position not in position_already_changed:
                    x = x[:position] + replacement + x[position+len(item):]
                    # update the changed position
                    position_already_changed_new = []
                    for pos in position_already_changed:
                        if pos < position:
                            position_already_changed_new.append(pos)
                        if pos > position:
                            position_already_changed_new.append(pos+len(replacement)-len(item))
                    for i in range(position, position + len(replacement)):
                        position_already_changed_new.append(i) 
                    # update the position in positions
                    c += 1
                    position_already_changed = position_already_changed_new
            dict_replacements['T' + str(index)] = item


    return x, dict_replacements

def string_to_z3_formula(formula_string, dict_var = dict_variable_symbol_object_universal, dict_const = dict_const_symbol_object_universal, dict_pred = dict_pred_symbol_object_universal):
    """
    Convert a string in Z3 Python syntax to a Z3 formula.
    Args:
        formula_string: String like 'Exists(x, Implies(And(U1(x), U2(z)), U3(z)))'
        dict_var: Dictionary mapping variable names to Z3 variable objects
        dict_const: Dictionary mapping constant names to Z3 constant objects
        dict_pred: Dictionary mapping predicate names to Z3 predicate objects
    Returns:
        Z3 formula object
    """
    
    
    # Create evaluation context
    context = {}
    
    # Add variables to context
    context.update(dict_var)
    
    # Add constants to context
    context.update(dict_const)
    
    # Add predicates to context
    context.update(dict_pred)
    
    # Add Z3 logical operators
    context.update({
        'And': And,
        'Or': Or, 
        'Not': Not,
        'Implies': Implies,
        'Exists': Exists,
        'ForAll': ForAll,
        'Eq': lambda x, y: x == y,  # For equality
        'True': True,
        'False': False,
    })
    
    # Parse the formula
    try:
        return eval(formula_string, {"__builtins__": {}}, context)
    except Exception as e:
        print(f"Error parsing formula string: {e}")
        return None
    

import re

def z3_to_logical_notation(formula):
    """
    Convert a Z3 formula to logical notation with symbols.
    
    Args:
        formula: Z3 formula object
    
    Returns:
        String in logical notation with ∃, ∀, ∧, ∨, →, ¬ symbols
    """
    if formula is None:
        return "None"

    if isinstance(formula, str):
        formula = string_to_z3_formula(formula)
    
    if isinstance(formula, QuantifierRef):
        if formula.is_exists():
            var_name = formula.var_name(0)
            body = formula.body().__str__()
            # replace in the body the right variable name
            for i in range(10):
                if f'Var({i})' in body:
                    body = body.replace(f'Var({i})', var_name)
            body_FOL = z3_to_logical_notation(body)
            return f"∃{var_name} {body_FOL}"
        
        if formula.is_forall():
            var_name = formula.var_name(0)
            body = formula.body().__str__()
            # replace in the body the right variable name
            for i in range(10):
                if f'Var({i})' in body:
                    body = body.replace(f'Var({i})', var_name)
            body_FOL = z3_to_logical_notation(body)
            return f"∀{var_name} {body_FOL}"
        
    
    op_name = formula.decl().name()
    
    if op_name == "and":
        args = ['(' + z3_to_logical_notation(formula.arg(i)) + ')' for i in range(formula.num_args())]
        return "(" + " ∧ ".join(args) + ")"
    
    elif op_name == "or":
        args = ['(' + z3_to_logical_notation(formula.arg(i)) + ')' for i in range(formula.num_args())]
        return "(" + " ∨ ".join(args) + ")"
    
    elif op_name == "=>":
        if formula.num_args() == 2:
            antecedent = z3_to_logical_notation(formula.arg(0))
            consequent = z3_to_logical_notation(formula.arg(1))
            return f"(({antecedent}) → ({consequent}))"
        else:
            return "→ (error)"
    
    elif op_name == "not":
        if formula.num_args() == 1:
            arg = z3_to_logical_notation(formula.arg(0))
            return f"¬{arg}"
        else:
            return "¬ (error)"
    
    elif op_name == "=":
        if formula.num_args() == 2:
            left = z3_to_logical_notation(formula.arg(0))
            right = z3_to_logical_notation(formula.arg(1))
            return f"({left} = {right})"
        else:
            return "= (error)"
    
    elif op_name == "iff":
        if formula.num_args() == 2:
            left = z3_to_logical_notation(formula.arg(0))
            right = z3_to_logical_notation(formula.arg(1))
            return f"({left} ↔ {right})"
        else:
            return "↔ (error)"
    
    else:
        # Atomic formula, predicate application, or unknown operator
        if formula.num_args() == 0:
            # Constant or variable
            return str(formula)
        else:
            # Function/predicate application
            func_name = formula.decl().name()
            args = [z3_to_logical_notation(formula.arg(i)) for i in range(formula.num_args())]
            return f"{func_name}({', '.join(args)})"


def push_negation_inside(formula, negate=False):
    """
    Convert a formula to Negation Normal Form (NNF).
    
    Args:
        formula: Z3 formula
        negate: Boolean indicating if we should negate the current formula
    
    Returns:
        Z3 formula in NNF where negation only appears on atomic formulas
    """
    
    
    if not negate:
        if isinstance(formula, QuantifierRef):
            if formula.is_exists():
                var_name = formula.var_name(0)
                var = dict_variable_symbol_object_universal[var_name]
                body = formula.body().__str__()
                # replace in the body the right variable name
                for i in range(10):
                    if f'Var({i})' in body:
                        body = body.replace(f'Var({i})', var_name)
                body_formula = string_to_z3_formula(body)
                return Exists(var, push_negation_inside(body_formula, False))

            if formula.is_forall():
                var_name = formula.var_name(0)
                var = dict_variable_symbol_object_universal[var_name]
                body = formula.body().__str__()
                # replace in the body the right variable name
                for i in range(10):
                    if f'Var({i})' in body:
                        body = body.replace(f'Var({i})', var_name)
                body_formula = string_to_z3_formula(body)
                return ForAll(var, push_negation_inside(body_formula, False))            

        # Get the operator name
        op_name = formula.decl().name()
        # No negation to push down
        if op_name == "and":
            # And(A, B) -> And(NNF(A), NNF(B))
            args = [push_negation_inside(formula.arg(i), False) for i in range(formula.num_args())]
            return And(*args)
        
        elif op_name == "or":
            # Or(A, B) -> Or(NNF(A), NNF(B))
            args = [push_negation_inside(formula.arg(i), False) for i in range(formula.num_args())]
            return Or(*args)
        
        elif op_name == "=>":
            # Implies(A, B) -> Or(Not(A), B) -> Or(NNF(~A), NNF(B))
            antecedent = push_negation_inside(formula.arg(0), True)  # Negate antecedent
            consequent = push_negation_inside(formula.arg(1), False)
            return Or(antecedent, consequent)
        
        elif op_name == "not":
            # Not(A) -> NNF(~A)
            return push_negation_inside(formula.arg(0), True)
        
        else:
            # Atomic formula or unknown operator - return as is
            return formula
    
    else:
        if isinstance(formula, QuantifierRef):
            if formula.is_exists():
                var_name = formula.var_name(0)
                var = dict_variable_symbol_object_universal[var_name]
                body = formula.body().__str__()
                # replace in the body the right variable name
                for i in range(10):
                    if f'Var({i})' in body:
                        body = body.replace(f'Var({i})', var_name)
                body_formula = string_to_z3_formula(body)
                return ForAll(var, push_negation_inside(body_formula, True))
                            
            if formula.is_forall():
                var_name = formula.var_name(0)
                var = dict_variable_symbol_object_universal[var_name]
                body = formula.body().__str__()
                # replace in the body the right variable name
                for i in range(10):
                    if f'Var({i})' in body:
                        body = body.replace(f'Var({i})', var_name)
                body_formula = string_to_z3_formula(body)
                return Exists(var, push_negation_inside(body_formula, True))
          
        # Get the operator name
        op_name = formula.decl().name()
        # We need to negate the current formula
        if op_name == "and":
            # ~And(A, B) -> Or(~A, ~B) (De Morgan's law)
            args = [push_negation_inside(formula.arg(i), True) for i in range(formula.num_args())]
            return Or(*args)
        
        elif op_name == "or":
            # ~Or(A, B) -> And(~A, ~B) (De Morgan's law)
            args = [push_negation_inside(formula.arg(i), True) for i in range(formula.num_args())]
            return And(*args)
        
        elif op_name == "=>":
            # ~Implies(A, B) -> ~Or(~A, B) -> And(~~A, ~B) -> And(A, ~B)
            antecedent = push_negation_inside(formula.arg(0), False)  # Don't negate
            consequent = push_negation_inside(formula.arg(1), True)   # Negate
            return And(antecedent, consequent)
        
        elif op_name == "not":
            # ~Not(A) -> A
            return push_negation_inside(formula.arg(0), False)
        
        else:
            # Atomic formula or unknown operator - apply negation
            return Not(formula)

def convert_to_nnf(formula):
    """
    Convert a formula to Negation Normal Form.
    
    Args:
        formula: Z3 formula
    
    Returns:
        Z3 formula in NNF
    """
    return push_negation_inside(formula, False)

def negate_to_nnf(formula):
    """
    Negate a formula and convert to Negation Normal Form.
    
    Args:
        formula: Z3 formula
    
    Returns:
        Negated formula in NNF
    """
    return push_negation_inside(formula, True)


def collect_subformulas(formula):
    """Collect all subformulas of a given formula"""
    subformulas = []
    
    def traverse(f):
        subformulas.append(f)
        if isinstance(f, QuantifierRef):
            # For quantified formulas, traverse the body
            body = f.body().__str__()
            var_name = f.var_name(0)
            for i in range(10):
                if f'Var({i})' in body:
                    body = body.replace(f'Var({i})', var_name)
            body = string_to_z3_formula(body)
            traverse(body)
        elif hasattr(f, 'children') and f.children():
            for child in f.children():
                traverse(child)

    traverse(formula)
    return subformulas

def find_subformula_at_position(formula, target_subformula):
    """Find the position of a subformula within the main formula"""

    if formula.eq(target_subformula):
        return []

    if isinstance(formula, QuantifierRef):
        var_name = formula.var_name(0)
        body = formula.body().__str__()
        for i in range(10):
            if f'Var({i})' in body:
                body = body.replace(f'Var({i})', var_name)
        body = string_to_z3_formula(body)
        path = find_subformula_at_position(body, target_subformula)
        if path is not None:
            return ['body'] + path
        
    if hasattr(formula, 'children') and formula.children():
        for i, child in enumerate(formula.children()):
            path = find_subformula_at_position(child, target_subformula)
            if path is not None:
                return [i] + path

    
    return None

def replace_subformula_at_path(formula, path, new_subformula):
    """Replace a subformula at a specific path with a new subformula"""
    if not path:
        return new_subformula
    
    if isinstance(formula, QuantifierRef):
        if path[0] == 'body':
            var_name = formula.var_name(0)
            body = formula.body().__str__()
            for i in range(10):
                if f'Var({i})' in body:
                    body = body.replace(f'Var({i})', var_name)
            body = string_to_z3_formula(body)
            new_body = replace_subformula_at_path(body, path[1:], new_subformula)
            if formula.is_exists():
                return Exists(dict_variable_symbol_object_universal[var_name], new_body)
            else:
                return ForAll(dict_variable_symbol_object_universal[var_name], new_body)
    elif hasattr(formula, 'children') and formula.children():
        children = list(formula.children())
        children[path[0]] = replace_subformula_at_path(children[path[0]], path[1:], new_subformula)
        return formula.decl()(*children)
    
    return formula

def apply_de_morgan(formula):
    """Apply De Morgan's law to a formula"""
    if is_not(formula):
        inner = formula.children()[0]
        if is_and(inner):
            # Not(And(a, b)) -> Or(Not(a), Not(b))
            return Or(*[Not(child) for child in inner.children()])
        elif is_or(inner):
            # Not(Or(a, b)) -> And(Not(a), Not(b))
            return And(*[Not(child) for child in inner.children()])
    elif is_and(formula) and (is_not(formula.children()[0]) and is_not(formula.children()[1])) :
        # And(a, b) -> Not(Or(Not(a), Not(b)))
        return Not(Or(formula.children()[0], formula.children()[1]))
    elif is_or(formula) and (is_not(formula.children()[0]) and is_not(formula.children()[1])):
        # Or(a, b) -> Not(And(Not(a), Not(b)))
        return Not(And(*[child for child in formula.children()]))
    
    return formula

def apply_distributivity(formula):
    """Apply distributivity law to a formula"""
    if is_and(formula):
        children = formula.children()
        for i, child in enumerate(children):
            if is_or(child):
                # And(Or(a, b), c) -> Or(And(a, c), And(b, c))
                other_children = [children[j] for j in range(len(children)) if j != i]
                or_children = child.children()
                return Or(*[And(or_child, *other_children) for or_child in or_children])
    elif is_or(formula):
        children = formula.children()
        for i, child in enumerate(children):
            if is_and(child):
                # Or(And(a, b), c) -> And(Or(a, c), Or(b, c))
                other_children = [children[j] for j in range(len(children)) if j != i]
                and_children = child.children()
                return And(*[Or(and_child, *other_children) for and_child in and_children])
    
    return formula

def apply_implication_expansion(formula):
    """Apply implication expansion to a formula"""
    if is_implies(formula):
        # Implies(a, b) -> Or(Not(a), b)
        children = formula.children()
        if is_not(children[0]):
            return Or(children[0], children[1])
        else:
            return Or(Not(children[0]), children[1])

    return formula

def apply_commutativity(formula):
    """Apply commutativity to a formula"""
    if is_and(formula) or is_or(formula):
        children = list(formula.children())
        children[0], children[1] = children[1], children[0]
        if is_and(formula):
            return And(*children)
        else:
            return Or(*children)
    
    return formula

def apply_double_negation(formula):
    """Apply double negation method for quantified formulas"""
    # Apply double negation: formula -> Not(Not(formula))
    # Then use negate_to_nnf on the inner negation
    inner_negation = negate_to_nnf(formula)
    return Not(inner_negation)

def generate_equivalent_formula(formula):
    """
    Generate an equivalent formula in the FOL format using random transformations
    """

    formula, dict_replacements = put_universal(formula)
    formula = parse_formula_SMTLIB_universal(formula)

    
    # For non-quantified formulas, apply propositional equivalences
    subformulas = collect_subformulas(formula)

    if not subformulas:
        # If no transformable subformulas, return original
        formula = z3_to_logical_notation(formula)
        for key, item in dict_replacements.items():
            formula = formula.replace(key, item)
        return formula
    

    found_target_subformula = False
    visited_subformulas = set()
    while (not found_target_subformula) and visited_subformulas != set(subformulas):
        # Randomly select a subformula to transform
        target_subformula = random.choice(subformulas)
        visited_subformulas.add(target_subformula)
        # Randomly select a transformation method
        transformations = []
        
        if isinstance(target_subformula, QuantifierRef):
            transformations.append(apply_double_negation)
        # Check which transformations are applicable
        if (is_not(target_subformula) and (is_and(target_subformula.children()[0]) or is_or(target_subformula.children()[0]))) or (is_and(target_subformula) and is_not(target_subformula.children()[0]) and is_not(target_subformula.children()[1])) or (is_or(target_subformula) and is_not(target_subformula.children()[0]) and is_not(target_subformula.children()[1])):
            transformations.append(apply_de_morgan)
        
        if is_and(target_subformula) or is_or(target_subformula):
            transformations.append(apply_commutativity)
        
        if (is_and(target_subformula) and is_or(target_subformula.children()[0])) or (is_or(target_subformula) and is_and(target_subformula.children()[0])):
            transformations.append(apply_distributivity)
        
        if is_implies(target_subformula):
            transformations.append(apply_implication_expansion)
        
        if is_eq(target_subformula):
            transformations.append(apply_commutativity)
        
        if transformations:
           break 

    if visited_subformulas == set(subformulas):
        formula = z3_to_logical_notation(formula)
        for key, item in dict_replacements.items():
            formula = formula.replace(key, item)
        return formula
            
    # Apply random transformation
    transformation = random.choice(transformations)
    # print(f'{transformation=}')
    transformed_subformula = transformation(target_subformula)
    
    # Replace the subformula in the original formula
    path = find_subformula_at_position(formula, target_subformula)
    if path is not None:
        result = replace_subformula_at_path(formula, path, transformed_subformula)
        result = z3_to_logical_notation(result)
        for key, item in dict_replacements.items():
            result = result.replace(key, item)
        return result
    
    formula = z3_to_logical_notation(formula)
    for key, item in dict_replacements.items():
        formula = formula.replace(key, item)
    return formula






# _____________
# _____________
#### Score evaluation metrics ######
# _____________
# _____________



























def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retry_with_exponential_backoff(func, max_retries  = 10, initial_delay: float = 0.2, max_delay: float = 15,
                                  backoff_factor: float = 1.2):
    """
    Retry a function with exponential backoff.
    """
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "limit" in str(e).lower() and "rate" in str(e).lower():
                    delay = min(max_delay, initial_delay * backoff_factor ** attempt)
                    time.sleep(delay)
                else:
                    raise e
    return wrapper

def return_rank(list_scores):
    ''' Given a list (or a list of list) of scores attributed to the objects, returns the list of the position numbers of the list_objects.

    RANKINGS starts with 0.
    '''

    def return_rank_single(list_scores):
        sorted_indices = sorted(range(len(list_scores)), key=lambda i: list_scores[i], reverse=True)
        list_position = [0 for item in list_scores]
        position = 0
        for i in range(len(sorted_indices)):
            list_position[sorted_indices[i]] = position
            if i < len(sorted_indices)-1 and list_scores[sorted_indices[i]] != list_scores[sorted_indices[i+1]]:
                position += 1
        return list_position
    
    if isinstance(list_scores[0], list):
        # case list of list
        return [return_rank_single(list_scores[i]) for i in range(len(list_scores))]
    else:
        return return_rank_single(list_scores)
    


def order_according_to_score(list_objects, list_scores):
    ''' Given a list (or a list of list) of objects and a list (or a list of list) of scores attributed to the objects, returns the list of objects ordered according to the scores in list_scores.
    '''
    def order_according_to_score_single(list_objects, list_scores):
        assert len(list_objects) == len(list_scores), "The list of the scores and the list of the objects must have the same length"
        sorted_indices = sorted(range(len(list_scores)), key=lambda i: list_scores[i], reverse=True)
        return [list_objects[i] for i in sorted_indices]
    
    if isinstance(list_scores[0], list):
        # case list of list
        assert len(list_objects) == len(list_scores), "The list of objects and the list of scores must have the same length"
        return [order_according_to_score_single(list_objects[i], list_scores[i]) for i in range(len(list_objects))]
    else:
        return order_according_to_score_single(list_objects, list_scores)


def shuffle_lists_take_k(list_objects, k):
    ''' Given a list (or a list of list) of objects, it returns a list (or a list of list) of maximum k objects randomly chosen from the list.

    objects in the list (or in the list of list) to shuffle are supposed to be not lists
    '''
    if len(list_objects) == 0:
        return None
    elif not isinstance(list_objects[0], list):
        return random.sample(list_objects, min(k, len(list_objects)))
    else:
        return [random.sample(list_objects[i], min(k, len(list_objects[i]))) for i in range(len(list_objects))]

def calculate_list_of_correct(list_scores, considered_position, place_ranking = 1):
    ''' Given a list of scores, it returns the accuarcy i.e. 1 for each time the object in the considered position comes at least at the right_place_ranking place in the ranking.

    The variable considered_position can be an integer or a list of integers. It can be an integer even in the case of a list of list of objects.
    
    '''
    


    def calculate_correct_single(list_scores, considered_position, place_ranking):
        assert considered_position < len(list_scores), "The considered position must be less than the length of the list of scores"
        rankings = return_rank(list_scores)
        return 1 if rankings[considered_position] <= place_ranking else 0

    is_list_of_list = isinstance(list_scores[0], list)
    if not is_list_of_list:
        return calculate_correct_single(list_scores, considered_position, place_ranking)
    else:
        if isinstance(considered_position, int):
            return [calculate_correct_single(list_scores[i], considered_position, place_ranking) for i in range(len(list_scores))]
        else:
            assert len(considered_position) == len(list_scores), "The list of considered positions and the list of scores must have the same length"
            return [calculate_correct_single(list_scores[i], considered_position[i], place_ranking) for i in range(len(list_scores))]

def compute_accuracy(list_of_correct):
    ''' Given a list of correct (1 if in the item the metric scored correctly, 0 otherwise), it returns the accuracy.'''
    return sum(list_of_correct)/len(list_of_correct)

def compute_info_wrong_answers(list_responses, list_of_correct):
    ''' Given a list of responses and a list of correct, it returns the information associated to wrong answers.
    
    list_responses is supposed to be a dictionary with two keys: info and answer'''
    assert len(list_responses) == len(list_of_correct), "The list of responses and the list of correct must have the same length"
    return [list_responses[i]['info'] for i in range(len(list_responses)) if list_of_correct[i] == 0]

def retrieve_info_from_custom_id(request_id):
    '''Retrieve the information from the request_id'''

    choice = request_id.split('_')[-2]
    number = request_id.split('_')[-1]
    if '-' in number:
        number_instance = number.split('-')[0]
        number_modification = number.split('-')[1]
        return choice, int(number_instance), int(number_modification)
    else:
        return choice, int(number)























# _____________
# _____________
##### saving and loading files ######
# _____________
# _____________






















def make_txt_into_list(directory_path):
    ''' Given a directory path, it returns a list of the files in the directory.'''

    assert directory_path.endswith('.txt'), "The directory path must end with .txt"
    with open(directory_path, 'r') as file:
        return [line.strip() for line in file]
    
def make_jsonl_into_list_dictionaries(directory_path):
    ''' Given a directory path, it returns a list of dictionaries.'''
    
    assert directory_path.endswith('.jsonl'), "The directory path must end with .jsonl"
    with open(directory_path, 'r') as file:
        return [json.loads(line) for line in file]
    
def make_list_dictionaries_into_jsonl(list_dictionaries, directory_path):
    ''' Given a list of dictionaries, it returns a jsonl file.'''
    
    assert directory_path.endswith('.jsonl'), "The directory path must end with .jsonl"
    with open(directory_path, 'w') as file:
        for dictionary in list_dictionaries:
            file.write(json.dumps(dictionary, ensure_ascii=False) + '\n')

def save_pickle(object, directory_path):
    ''' Given an object, it returns a pickle file.'''
    
    assert directory_path.endswith('.pkl'), "The directory path must end with .pkl"
    with open(directory_path, 'wb') as file:
        pickle.dump(object, file)
    
def load_pickle(directory_path):
    ''' Given a directory path, it returns a pickle file.'''
    
    assert directory_path.endswith('.pkl'), "The directory path must end with .pkl"
    with open(directory_path, 'rb') as file:
        return pickle.load(file)
    

def from_list_combo_to_transl_name(list_combo):
    '''Given the list of combinations ([0,0,0,1,0]), returns the name of the associated file ('Transl_(0,0,0,1,0))'''
    s = '('
    for i in range(len(list_combo)-1):
        s += str(list_combo[i])
        s += ', '
    s += str(list_combo[-1]) + ')'
    return f'Transl_{s}.pkl'



















    
















###### OpenAI API ######








def create_openai_list_requests_for_batch_embeddings(model, custom_id : str, list_sentences_to_embed : list[str],  method : str = 'POST') -> list[dict]:
    ''' Returns the list of the requests for an openAI call of the model in the batch version.

    custom_id is used to identify the request in the batch: the number of the instance in the list will be append.
    '''

    list_requests = []
    for i, sentence in enumerate(list_sentences_to_embed):
        list_requests.append({
            "custom_id": f"{custom_id}-{i}",
            "method": method,
            "url": "/v1/embeddings",
            "body": {
                "model": model,
                "input": sentence,
            }
        })
    return list_requests
    

def create_openai_list_requests_for_batch_completions(model, custom_id : str,  system_prompt : str|list[str] | dict, user_prompts : list[str] | dict, response_format, seed: int = None, max_tokens = 1500,  max_context_tokens = 10000) -> list[dict]:
    ''' Returns the list of the requests for an openAI call of the model in the batch version.

    custom_id is used to identify the request in the batch: the number of the instance in the list will be appended.
    
    system_prompt and list_user_prompts are lists of the same length or system_prompt is a string supposing that the same system prompt is used for all the requests.

    max_context_tokens is the maximum number of tokens in the context, hence, it will raise error if the context is too long.

    response_format is a type_to_response_format(pydantic model) (json)
    '''

    list_requests = []

    if isinstance(user_prompts, list):
        assert isinstance(system_prompt, list) or isinstance(system_prompt, str)
        #assert len(system_prompt) == len(user_prompts), "The system prompt and the list of user prompts must have the same length"

        for i, user_prompt_i in enumerate(user_prompts):
            system_prompt_i = system_prompt[i] if isinstance(system_prompt, list) else system_prompt
            assert len(system_prompt_i) + len(user_prompt_i) <= max_context_tokens, "The context is too long"
            if 'o3' in model:
                body = {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt_i}, {"role": "user", "content": user_prompt_i}],
                    "max_completion_tokens": max_tokens,
                }
            else:
                body = {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt_i}, {"role": "user", "content": user_prompt_i}],
                    "max_tokens": max_tokens,
                }
            if response_format is not None:
                body['response_format'] = response_format
            
            if seed is not None:
                body['seed'] = seed

            list_requests.append({
                "custom_id": f"{custom_id}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            })

        return list_requests
    
    if isinstance(user_prompts, dict):
        assert isinstance(system_prompt, dict) or isinstance(system_prompt, str)
        for key in user_prompts.keys():
            system_prompt_i = system_prompt[key] if isinstance(system_prompt, dict) else system_prompt
            user_prompt_i = user_prompts[key]
            assert len(system_prompt_i) + len(user_prompt_i) <= max_context_tokens, "The context is too long"
            if 'o3' in model:
                body = {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt_i}, {"role": "user", "content": user_prompt_i}],
                    "max_completion_tokens": max_tokens,
                }
            else:
                body = {
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt_i}, {"role": "user", "content": user_prompt_i}],
                    "max_tokens": max_tokens,
                }
            if response_format is not None:
                body['response_format'] = response_format

            if seed is not None:
                body['seed'] = seed


            list_requests.append({
                "custom_id": f"{custom_id}_{key}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            })
        return list_requests
    
    





def call_openai_api_batch_completions(list_requests : list[dict], directory_path : str, file_name : str) -> list[dict]:
    ''' Given a list of requests, it calls the openAI API in the batch version and returns the batch id and the batch object.

    directory_path is the folder where the requests file is saved.
    file_name is the name of the file where the requests are saved (.jsonl)
    '''

    assert file_name.endswith('.jsonl'), "The file name must end with .jsonl"
    assert os.path.isdir(directory_path), "The directory path must be a valid directory"

    load_dotenv()

    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key in the .env file"

    openai_client = OpenAI()

    print("DEBUG: list_requests: ", list_requests)
    file_path = os.path.join(directory_path, file_name)
    make_list_dictionaries_into_jsonl(list_requests, file_path)

    # Create the batch input file
    batch_input = openai_client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    batch_input_id = batch_input.id
    batch = openai_client.batches.create(
        input_file_id=batch_input_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_id = batch.id
    batch = openai_client.batches.retrieve(batch_id)

    return batch_id, batch

def call_openai_api_batch_embeddings(list_requests : list[dict], directory_path : str, file_name : str):
    ''' Given a list of requests, it calls the openAI API in the batch version and returns the batch id and the batch object.

    directory_path is the folder where the requests file is saved.
    file_name is the name of the file where the requests are saved (.jsonl)
    '''

    assert file_name.endswith('.jsonl'), "The file name must end with .jsonl"
    assert os.path.isdir(directory_path), "The directory path must be a valid directory"

    load_dotenv()

    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key in the .env file"

    openai_client = OpenAI()


    file_path = os.path.join(directory_path, file_name)
    make_list_dictionaries_into_jsonl(list_requests, file_path)

    # Create the batch input file
    batch_input = openai_client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )
    batch_input_id = batch_input.id
    batch = openai_client.batches.create(
        input_file_id=batch_input_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
    )
    batch_id = batch.id
    batch = openai_client.batches.retrieve(batch_id)

    return batch_id, batch

def extract_batch_errors_openai(batch_object):
    '''Given a batch object, it returns the list of the errors of the batch, if some exist.'''
    load_dotenv()

    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key in the .env file"

    openai_client = OpenAI()

    # assert batch_object.status == 'completed', "Batch call not yet completed"

    if batch_object.error_file_id:
        errors = openai_client.files.content(batch_object.error_file_id).content.decode('utf-8')
        return [json.loads(line) for line in errors.strip().split('\n')]
    else:
        return None

def extract_batch_outputs_openai(batch_object):
    ''' Given a batch object, it returns the list of the responses of the batch.'''
    load_dotenv()

    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key in the .env file"

    openai_client = OpenAI()

    # assert batch_object.status == 'completed', "Batch call not yet completed"

    if batch_object.output_file_id:
        outputs = openai_client.files.content(batch_object.output_file_id).content.decode('utf-8')
        return [json.loads(line) for line in outputs.strip().split('\n')]
    else:
        return None



def extract_batch_errors_output_openai(batch_object) -> list[dict]:
    ''' Given a batch object, it returns the list of the responses of the batch.

    batch_object is the batch object.
    '''
    

    load_dotenv()

    assert os.getenv("OPENAI_API_KEY") is not None, "You must set your OpenAI API key in the .env file"

    openai_client = OpenAI()

    if batch_object.error_file_id:
        errors = openai_client.files.content(batch_object.error_file_id).content.decode('utf-8')
        print('Error file: ')
        return [json.loads(line) for line in errors.strip().split('\n')]
    else:
        print('No error file available for this batch.\n')
        if batch_object.output_file_id:
            outputs = openai_client.files.content(batch_object.output_file_id).content.decode('utf-8')
            print('Output file: ')
            return [json.loads(line) for line in outputs.strip().split('\n')]
        else:
            print('No output file available for this batch.')
            return []


def compute_dictionary_evaluation_prompting(list_LLM_answer, list_correct_answer):
    '''Given the list (of a list) of the LLM answers, it returns  a dictionary with the following keys:
    - list_correct_answers (list of 0/1 depending on the correctness of the LLM answer);
    - accuracy
    - info_wrong (reasoning in the case of wrong answers. Is a dictionary with key the number of instance considered)'''

    def compute_single_dict_eval_prompt(list_LLM_answer, list_correct_answer):
        list_correct = [0 for i in range(len(list_LLM_answer))]
        for i, item in enumerate(list_LLM_answer):
            if item['answer'] == list_correct_answer[i]:
                list_correct[i] = 1
        

    
    if isinstance(list_LLM_answer[0],list):
        dict = []
    else:
        list_correct = compute_single_dict_eval_prompt(list_LLM_answer, list_correct_answer)






########### Google API ###########






def create_google_list_requests_for_batch_embeddings(model, custom_id : str, list_sentences_to_embed : list[str], method : str = 'POST') -> list[dict]:
    ''' Returns the list of the requests for an openAI call of the model in the batch version.

    custom_id is used to identify the request in the batch: the number of the instance in the list will be append.
    '''

    pass


    # print('Not implemented yet')
    # return []

    # list_requests = []
    # for i, sentence in enumerate(list_sentences_to_embed):
    #     list_requests.append({
    #         "custom_id": f"{custom_id}_{i}",
    #         "method": method,
    #         "url": "/v1/embeddings",
    #         "body": {
    #             "model": model,
    #             "input": sentence,
    #         }
    #     })
    # return list_requests


















# _____________
# _____________
### Test sensitivity ###
# _____________
# _____________






def plotting_feature_relevance(list_scores_without : list[float], list_scores_with : list[float], list_name_measures : list[str] = None, figsize = (16,12), style = 'grouped'):
    '''Returns a bar diagram that shows the relevance of a feature'''
    assert len(list_scores_with) == len(list_scores_without)
    if list_name_measures:
        assert len(list_scores_with) == len(list_name_measures), "list_name_measures should be as long as list_scores"
    else:
        list_name_measures = ['' for i in range(list_scores_with)]


    def _plot_grouped_bars(ax, measures, M_without, M_with):
        """Helper function to create grouped bar chart"""
        x = np.arange(len(measures))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, M_without, width, label='Without Feature', alpha=0.8, color='steelblue')
        bars2 = ax.bar(x + width/2, M_with, width, label='With Feature', alpha=0.8, color='coral')
        
        ax.set_xlabel('Measures')
        ax.set_ylabel('Values')
        ax.set_title('Grouped Bar Chart: Feature Impact Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(measures, rotation=45 if len(max(measures, key=len)) > 3 else 0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(M_without)*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(M_with)*0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    def _plot_difference_chart(ax, measures, M_without, M_with):
        """Helper function to create difference chart"""
        differences = M_with - M_without
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        bars = ax.bar(measures, differences, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Measures')
        ax.set_ylabel('Change (With - Without)')
        ax.set_title('Feature Impact: Direct Change Visualization')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-labels if needed
        if len(max(measures, key=len)) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            offset = abs(max(differences) - min(differences)) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2., 
                    height + (offset if height > 0 else -offset),
                    f'{diff:+.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)

    def _plot_connected_dots(ax, measures, M_without, M_with):
        """Helper function to create connected dot plot"""
        for i, measure in enumerate(measures):
            ax.plot([i, i], [M_without[i], M_with[i]], 'o-', 
                    color='gray', alpha=0.6, linewidth=2, markersize=8)
            ax.scatter(i, M_without[i], color='steelblue', s=80, label='Without Feature' if i == 0 else "", zorder=5)
            ax.scatter(i, M_with[i], color='coral', s=80, label='With Feature' if i == 0 else "", zorder=5)
        
        ax.set_xlabel('Measures')
        ax.set_ylabel('Values')
        ax.set_title('Before/After Comparison: Connected Dot Plot')
        ax.set_xticks(range(len(measures)))
        ax.set_xticklabels(measures, rotation=45 if len(max(measures, key=len)) > 3 else 0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_percentage_change(ax, measures, M_without, M_with):
        """Helper function to create percentage change chart"""
        # Handle division by zero
        percent_change = np.where(M_without != 0, ((M_prime - M_without) / M_without) * 100, 0)
        colors = ['green' if p > 0 else 'red' for p in percent_change]
        
        bars = ax.bar(measures, percent_change, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_xlabel('Measures')
        ax.set_ylabel('Percentage Change (%)')
        ax.set_title('Feature Impact: Percentage Change')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-labels if needed
        if len(max(measures, key=len)) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, pct in zip(bars, percent_change):
            height = bar.get_height()
            offset = abs(max(percent_change) - min(percent_change)) * 0.05
            ax.text(bar.get_x() + bar.get_width()/2., 
                    height + (offset if height > 0 else -offset),
                    f'{pct:+.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)


    # Convert to numpy arrays for easier manipulation
    M_without = np.array(list_scores_without)
    M_with = np.array(list_scores_with)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    if style == 'all':
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=figsize)
        
        # 1. GROUPED BAR CHART
        ax1 = plt.subplot(2, 2, 1)
        _plot_grouped_bars(ax1, list_name_measures, M_without, M_with)
        
        # 2. DIFFERENCE/DELTA CHART
        ax2 = plt.subplot(2, 2, 2)
        _plot_difference_chart(ax2, list_name_measures, M_without, M_with)
        
        # 3. CONNECTED DOT PLOT
        ax3 = plt.subplot(2, 2, 3)
        _plot_connected_dots(ax3, list_name_measures, M_without, M_with)
        
        # 4. PERCENTAGE CHANGE CHART
        ax4 = plt.subplot(2, 2, 4)
        _plot_percentage_change(ax4, list_name_measures, M_without, M_with)
        
        plt.tight_layout()
        
    else:
        # Create single plot
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]//2))
        
        if style == 'grouped':
            _plot_grouped_bars(ax, list_name_measures, M_without, M_with)
        elif style == 'difference':
            _plot_difference_chart(ax, list_name_measures, M_without, M_with)
        elif style == 'connected':
            _plot_connected_dots(ax, list_name_measures, M_without, M_with)
        elif style == 'percentage':
            _plot_percentage_change(ax, list_name_measures, M_without, M_with)
        else:
            raise ValueError("style must be one of: 'all', 'grouped', 'difference', 'connected', 'percentage'")
    
    return fig







# def test_sensitivity(measure_names, list_file_names):
#     for measure in measure_names:
#         if measure == 'text-embedding-3-small':
#             dict