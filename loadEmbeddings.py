from config import *
from Embeddings import Embeddings
import torch
import argparse
from embedding_config import load_hyperparam
import bert_encoder
from transformers import BertTokenizer, BertModel
import numpy as np

# give all words in sentence and target unique count
all_whole_sentence = []
all_counted_tokens = []
count = -1
count2 = 1
string_year = str(FLAGS.year)
use_vm = False
softpositions = False

if FLAGS.indicator_sentence == '_normal':
    use_vm = True
    softpositions = True
elif FLAGS.indicator_sentence =='_without_softpos':
    use_vm = True
    softpositions = False
elif FLAGS.indicator_sentence =='_without_vm':
    use_vm = False
    softpositions = True


model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()

# Create the embeddings and data with added words
emb = Embeddings(sofpos=softpositions, vm=use_vm)
emb.makeEmbeddings()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--config_path", default="./google_config.json", type=str,help="Path of the config file.")
args = parser.parse_args()
args = load_hyperparam(args) # Load the hyperparameters from the config file.

encoder = bert_encoder.BertEncoder(args, model) # make object of bert_encoder

tokens = emb.sentences # list of list of all tokens for a sentence
vm = emb.vm_tensors # a list of visible matrices for each sentence in tensor-shape of 1*token_numb*token_numb.
embeddings = emb.embeddings # a list with intial embeddings for each token in tensor-shape of 1*token_numb*768
hidden_states = []  # list of hidden states
token_hidden_states = [] # list with for each token, the token and the hidden states

count = 0
end = FLAGS.end

# Calculate the hidden state of each token
for i in range(len(emb.sentences)):
    if use_vm:
        # calculate all hidden states for all tokens in a sentence
        hidden = encoder.forward(embeddings[i], None, vm[i])
    else:
        # calculate all hidden states for all tokens in a sentence, without visible matrix
        tensor = torch.zeros((1, len(emb.soft_positions[i]), len(emb.soft_positions[i])))
        hidden = encoder.forward(embeddings[i], None, tensor)
    hidden_states.append(hidden)


# Create a txt file that includes a unique count for each token, together with its embedding
dictionary = dict()
if FLAGS.start > 0:
    dictionary = np.load(f'Dictionary{FLAGS.year}{FLAGS.indicator_sentence}_{FLAGS.start}_H={FLAGS.hops}.npy', allow_pickle='TRUE').item()
with open('data/programGeneratedData/' + f'allEmbeddings{FLAGS.year}{FLAGS.indicator_sentence}_H={FLAGS.hops}_{FLAGS.end}.txt','w') as outf:
    for j in range(len(emb.sentences)):
        print('{}/{}'.format(j, len(emb.sentences)))
        token_count = 0
        for token in emb.sentences[j]: # iterate over all tokens in a sentence
            if token == "[CLS]" or token == "[SEP]":
                token_count += 1
            else:
                list_of_embeddings = hidden_states[j][0][token_count].tolist() # make a list of the embedding per token
                token_count += 1 # count the tokens
                if not token in dictionary:
                    dictionary[token] = 0
                else:
                    past_value = dictionary[token]
                    dictionary[token] = past_value + 1
                token = token + '_' + str(dictionary[token])
                string_list_of_embeddings = []
                string_list_of_embeddings = [str(round(i, 8)) for i in list_of_embeddings]  # convert numbers to strings
                string_list_of_embeddings.insert(0, token)  # append the token in the front of the list
                print(" ".join(string_list_of_embeddings), file=outf)
                # token_hidden_states.append(string_list_of_embeddings)  # append the embedding for a word to the big list

np.save(f'Dictionary{FLAGS.year}{FLAGS.indicator_sentence}_{FLAGS.end}_H={FLAGS.hops}.npy', dictionary)