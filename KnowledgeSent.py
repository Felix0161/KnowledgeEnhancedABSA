import numpy as np
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from config import *
from KnowledgeBranch import KnowledgeBranch
from transformers import BertTokenizer


class KnowledgeSentence():
    '''
    Transforms input sentence into a sentence tree by appending knowledge branches
    '''

    def __init__(self, sent, hops, tokenizer, include_knowledge=True):
        self.sent = sent.split(' ')
        self.hops = hops
        self.tokenizer = tokenizer
        self.original_ids = []
        self.original = []
        self.sentence = []


        if include_knowledge:
            self.sentence, self.original, self.original_ids, self.soft_positions, self.visibility_matrix = self.get_sentence()
        else:
            for w in self.sent:
                s = ' '.join(self.tokenizer.tokenize(w))
                if '$ t $' in s:
                    s = s.replace('$ t $', '$T$')
                self.sentence += s.split(' ')
            self.original += [1] * len(self.sentence)
            for i in range(len(self.sentence)):
                self.original_ids.append(i)
            self.visibility_matrix = np.ones((len(self.sentence), len(self.sentence)))
            self.soft_positions = self.original_ids
            self.visibility_matrix = self.visibility_matrix * 10000
            self.visibility_matrix = self.visibility_matrix - 10000

        self.segments = self.makeSegments()

    def get_sentence(self):
        sentence = self.sent
        tok_sent = self.tokenizer.tokenize(' '.join(sentence))
        s = ' '.join(tok_sent)
        if '$ t $' in s:
            s = s.replace('$ t $', '$T$')
        tok_sent = s.split(' ')
        original = []
        original_ids = []
        soft_positions = []
        visibility = []
        extra_room = FLAGS.max_sentence_len - len(sentence)


        know_sent = []
        ori_id = 0
        soft_p = 0
        if len(tok_sent) > FLAGS.max_sentence_len:
            sentence = tok_sent[:FLAGS.max_sentence_len]
            original = np.ones(len(sentence))
            original_ids = np.arange(len(sentence))
            soft_positions = original_ids
            visibility_matrix = np.ones((len(sentence),len(sentence)))
        else:
            for token in tok_sent:
                original_ids.append(ori_id)
                original.append(1)
                if extra_room > 0:
                    b = KnowledgeBranch(token, self.hops, soft_p, self.tokenizer, max_branch_length = extra_room + 1)
                    extra_room = extra_room - len(b.divided_token_tree) + 1
                else:
                    b = KnowledgeBranch(token, self.hops, soft_p, self.tokenizer, max_branch_length = 1)
                for row in b.token_matrix:
                    visibility.append(row.copy())
                know_sent += b.divided_token_tree
                original += [0] * (len(b.divided_token_tree) - 1)
                soft_positions += b.soft_pos
                ori_id += len(b.divided_token_tree)
                soft_p += 1

            sentence = know_sent
            visibility_matrix = np.zeros((len(sentence), len(sentence)))
            for i in range(len(original_ids)):
                visibility_matrix[original_ids[i], original_ids] = 1
                try:
                    r = range(original_ids[i], original_ids[i + 1])
                    for j in r:
                        row = visibility[j]
                        visibility_matrix[j][range(original_ids[i], original_ids[i] + len(r))] = row
                except IndexError:
                    break

        visibility_matrix = visibility_matrix*10000
        visibility_matrix = visibility_matrix - 10000
        return sentence, original, original_ids, soft_positions, visibility_matrix

    def makeSegments(self):
        """
        determines the segment id's
        """
        seg = []
        s_count = 0

        for token in self.sentence:
            if s_count == 0 or s_count % 2 == 0:
                seg.append(0)  # can change it to zero or 1
            elif s_count == 1 or s_count % 2 != 0:
                seg.append(1)
            # if token == "-" or token=='–': #remove hashes to alternate between segments for minus sign or dash
            #     s_count += 1
        return seg


