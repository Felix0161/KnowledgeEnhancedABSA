import numpy as np
from config import *
from KnowledgeSent import KnowledgeSentence
from transformers import BertTokenizer, BertModel
import torch

hops = FLAGS.hops
string_year = str(FLAGS.year)
number = 0
start = FLAGS.start
end = FLAGS.end
if string_year == '2015':
    number = 3864
elif string_year == '2016':
    number = 5640


class Embeddings(object):
    '''
    For each input sentence:
    - Token embedding of each token
    - Segment embedding
    - Position embedding
    - Visibility matrix
    '''

    def __init__(self, sofpos=True, vm=True):
        self.sentences = []  # list of list of all tokens for a sentence
        self.visible_matrices = []
        self.vm_tensors = [] # a list of visible matrices for each sentence in tensor-shape of 1*token_numb*token_numb.
        self.soft_positions = [] # list of list of softpositions for each sentence
        self.segments = [] # list of list of segments for each sentence
        self.embeddings = []  # a list with intial embeddings for each token in tensor-shape of 1*token_numb*768
        self.hidden_states = []  # list of hidden states
        self.token_hidden_states = []  # list with for each token, the token and the hidden states
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.sofpos = sofpos
        self.vm = vm

    def makeEmbeddings(self):
        count = 0
        with open('data/externalData/' + 'raw_data' + string_year + '.txt', 'r') as raw_data:
            line_list = raw_data.readlines()
            for i in range(start*3, min(number, end*3)):
                if i % 600 == 0:
                    print('another 200, ({}/{})'.format(i, min(number, end*3)))
                if count % 3 == 0 :  # if it is a sentence line
                    # add CLS and SEP and remove target sign
                    sentence = "[CLS] " + line_list[i].replace('$T$', line_list[i+1].replace('\n', '')) + " [SEP]"
                    # add no knowledge
                    sent = KnowledgeSentence(sentence, hops, self.tokenizer, include_knowledge=False)
                    self.sentences.append(sent.sentence)
                    self.soft_positions.append(sent.soft_positions)
                    self.segments.append(sent.segments)
                    self.visible_matrices.append(sent.visibility_matrix)
                else:  # if it is a target line or sentiment line
                    pass
                count += 1
            # append the raw test data and add knowledge
            for i in range(max(number, start*3), end*3):
                if i % 600 == 0:
                    print('another 200 ({}/{})'.format(i, end*3))
                if count % 3 == 0: # if it is a sentence line
                    # add CLS and SEP and replace $T$-token with target-token
                    sentence = "[CLS] " + line_list[i].replace('$T$', line_list[i+1].replace('\n', '')) + " [SEP]"
                    # add knowledge to sentence
                    know_sent = KnowledgeSentence(sentence, hops, self.tokenizer, include_knowledge=True)
                    self.sentences.append(know_sent.sentence)
                    self.soft_positions.append(know_sent.soft_positions)
                    self.segments.append(know_sent.segments)
                    if self.vm:
                        self.visible_matrices.append(know_sent.visibility_matrix)
                    else:
                        self.visible_matrices.append(np.ones((len(know_sent.sentence), len(know_sent.sentence))))
                else: # if it is a target line or sentiment line
                    pass
                count += 1

        print('Creating embeddings...')
        for i in range(len(self.sentences)):
            token_tensor = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.sentences[i])])
            segment_tensor = torch.tensor([self.segments[i]])
            pos_tensor = torch.tensor([self.soft_positions[i]])
            if self.sofpos:
                output = self.model(token_tensor, None, segment_tensor, pos_tensor)
            else:
                output = self.model(token_tensor, None, segment_tensor, None)
            tensor = output.hidden_states.__getitem__(00)

            self.embeddings.append(tensor)
            self.vm_tensors.append(torch.tensor(self.visible_matrices[i]))
        print('Embeddings created!')



