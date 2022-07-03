from owlready2 import *
import numpy as np
import config
from transformers import BertTokenizer

class KnowledgeBranch():
    '''
    Creates a knowledge branch which is to be added to the input word, as well as its corresponding visibility matrix and softpositions
    '''

    def __init__(self, word_token, hops, pos, tokenizer, max_branch_length =config.FLAGS.max_sentence_len, include_superclasses=True, include_subclasses=True, max_entities=config.MAX_ENTITIES):
        onto_path.append("data/externalData")  # path
        self.onto = get_ontology("ontology.owl-expanded.owl").load()
        self.classes = set(self.onto.classes())
        self.dictionary = {}
        self.synonym_dict = {}
        for onto_class in self.classes:
            self.synonym_dict[onto_class] = onto_class.lex.copy()
            for tok in onto_class.lex:
                self.dictionary[tok] = onto_class
        self.word = word_token
        self.pos = pos - 1
        self.soft_pos = []
        self.max_pos = pos
        self.word_tree = []
        self.token_tree = []
        self.divided_token_tree = []
        self.soft_dict = {}
        self.hard_dict = {}
        self.max_entities = max_entities
        self.tokenizer = tokenizer



        if not (include_superclasses or include_subclasses):
            self.word_tree.append(self.word)
            self.syn = self.get_synonyms(self.word)[:self.max_entities]
            self.word_tree.extend(self.syn)
            self.soft_dict[self.word] = self.syn

        else:
            if include_superclasses:
                self.word_superclasses = self.hop(self.word, hops, self.pos, direction=True)
            if include_subclasses:
                self.word_subclasses = self.hop(self.word, hops, self.pos, direction=False)
        self.visibility_matrix, self.token_matrix = self.get_visible_words()

        if len(self.divided_token_tree) > max_branch_length:
            self.divided_token_tree = self.divided_token_tree[:max_branch_length]
            self.visibility_matrix = self.visibility_matrix[0:max_branch_length,0:max_branch_length]
            self.token_matrix = self.token_matrix[0:max_branch_length,0:max_branch_length]
            self.soft_pos = self.soft_pos[:max_branch_length]




    def hop(self, word, hops, pos, direction=True, ancestors=set()):
        '''
        Recursive function for creating a knowledge branch.
            input:
            word - word to which knowledge branch is appended
            hops - depth of the knowledge branch
            pos - softposition of the word in the sentence or branch


            return: tup - tuple with following structure: {word, synonyms, tuples of related words}
        '''

        syn = self.get_synonyms(word)
        new_pos = pos + 1
        if new_pos > self.max_pos:
            self.max_pos = new_pos

        if word in self.hard_dict.keys():
            ancestors.clear()
        else:
            self.word_tree.append(word)
            self.hard_dict[word] = set()
            self.word_tree.extend(syn)
            self.soft_dict[word] = syn
            self.soft_pos +=[new_pos]
            self.soft_pos += [new_pos + 1]*len(syn)



        tup = (word, syn, [])

        try:
            key = self.dictionary[word.capitalize()]
        except KeyError:
            try:
                key = self.dictionary[word.lower()]
            except KeyError:
                return tup
        if direction:
            entities = key.is_a
        else:
            entities = [c for c in self.onto.get_children_of(key)]
        if hops > 0 and len(entities) > 0:
            entities = entities[:self.max_entities]
            for i in range(len(entities)):
                ent = entities[i].lex
                try:
                    ent = ent[0]
                except IndexError:
                    break
                anc = set(ancestors)
                for a in anc:
                    self.hard_dict[a].add(ent)
                self.hard_dict[word].add(ent)
                ancestors.add(word)
                tup[2].append(self.hop(ent, hops - 1, new_pos, direction, ancestors)) # A branch of k hops for word i is equal to the collection of branches of k-1 hops of word i's children
        else:
            ancestors.clear()
        return tup

    def get_synonyms(self,w):
        try:
            key = self.dictionary[w.capitalize()]
        except KeyError:
            try:
                key = self.dictionary[w.lower()]
            except KeyError:
                return []
        syn = self.synonym_dict[key].copy()

        try:
            syn.remove(w.lower())
        except ValueError:
            try:
                syn.remove(w.capitalize())
            except ValueError:
                pass

        return syn[:self.max_entities]

    def make_matrices(self):
        original_soft_pos = self.soft_pos.copy()
        count = 0
        for i in range(len(self.word_tree)):
            wrd = self.word_tree[i]
            if wrd == '$T$' or wrd.startswith('##'):
                tok = [wrd]
            else:
                tok = self.tokenizer.tokenize(wrd)
            self.token_tree.append(tok)
            self.divided_token_tree += tok
            if len(tok) > 1:
                self.soft_pos[i+1+count:i+1+count] = np.arange(original_soft_pos[i] + 1, original_soft_pos[i] + len(tok))
                count += len(tok) - 1
        token_matrix = np.zeros((len(self.divided_token_tree), len(self.divided_token_tree)))
        visibility_matrix = np.zeros((len(self.word_tree), len(self.word_tree)))
        return visibility_matrix, token_matrix

    def get_visible_words(self):
        '''
        Function to determine which words are visible to each other. Used for creating the visibility matrices.
        '''

        visible_words = []
        visibility_matrix, token_matrix = self.make_matrices()
        i = 0
        for key in self.hard_dict.keys():
            own_id = self.word_tree.index(key)
            visibility_matrix[own_id][own_id] = 1
            for child in self.hard_dict[key]:
                self.hard_dict[child].add(key)
            v_words = self.hard_dict[key]

            for syn in self.get_synonyms(key):
                v_words.add(syn)
                visibility_matrix[self.word_tree.index(syn), own_id] = 1
                visibility_matrix[self.word_tree.index(syn), self.word_tree.index(syn)] = 1


            visible_words.append(v_words)
            visible_ids = []
            for vw in v_words:
                ids = self.word_tree.index(vw)
                visible_ids.append(ids)
            visibility_matrix[self.word_tree.index(key), visible_ids] = 1

        added_number = 0
        for row_index in range(len(visibility_matrix[0])):
            row = visibility_matrix[row_index]
            token_row = []
            number_of_tokens = 0
            for r in range(len(row)):
                token_row += [row[r]]*len(self.token_tree[r])
            rows_to_add = len(self.token_tree[row_index])
            while number_of_tokens < rows_to_add:
                token_matrix[row_index + added_number + number_of_tokens] = token_row
                number_of_tokens += 1
            if rows_to_add > 1:
                added_number += rows_to_add-1
        return visibility_matrix, token_matrix