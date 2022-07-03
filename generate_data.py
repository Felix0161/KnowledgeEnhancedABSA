from KnowledgeSent import*
from transformers import BertTokenizer
from config import *


def divide_words_in_sentence(tokenizer, sent):
    """
    returns a tokenized sentence and makes sure that target sign isn't tokenized
    """
    list_with_dividing = tokenizer.tokenize(sent)
    sentence = ' '.join(list_with_dividing)
    if '$ t $' in sentence:
        sentence2 = sentence.replace('$ t $', '$T$')
    else:
        sentence2 = sentence
    return sentence2


class TestData:

    # determine line where test data starts
    number = 0
    if FLAGS.year == 2016:
        number = 5640
    elif FLAGS.year == 2015:
        number = 3864


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenize the raw data
    # and append synonyms on the test part of the data
    with open('data/externalData/' + 'raw_data' + str(FLAGS.year) + '.txt', 'r') as raw_data:
        line_list = raw_data.readlines()
        with open('data/temporaryData/' + 'data' + str(FLAGS.year) + '_H=' + str(
                FLAGS.hops) + '.txt',
                  'w') as test_data:
            for i in range(0, len(line_list), 3):
                s = line_list[i]
                s = s.replace('\n', '')
                target = line_list[i + 1]
                target = target.replace('\n', '')
                sentiment = str(line_list[i + 2])
                if i >= number:
                    s = KnowledgeSentence(s, FLAGS.hops, tokenizer)
                    t = KnowledgeSentence(target, FLAGS.hops, tokenizer)
                    try: # only add target knowledge, not target subtokens
                        s.sentence[s.sentence.index('$T$') + 1:s.sentence.index('$T$') + 1] = t.sentence[len(tokenizer.tokenize(
                            line_list[i + 1])):]
                    except ValueError:
                        pass
                else:
                    s = KnowledgeSentence(s, FLAGS.hops, tokenizer, include_knowledge=False)
                    t = KnowledgeSentence(target, FLAGS.hops, tokenizer, include_knowledge=False)

                s = ' '.join(s.sentence)
                t = ' '.join(t.sentence)

                test_data.write(s + '\n')
                test_data.write(target + '\n')
                test_data.write(str(sentiment))

    raw_data.close()
    test_data.close()


    all_whole_sentence = []
    count = -1
    count2 = 1
    # make the tokens in the data unique
    with open('data/temporaryData/' + 'data' + str(FLAGS.year) + '_H=' + str(FLAGS.hops) + '.txt', 'r') as test_data1:
        line_list = test_data1.readlines()
        for i in range(0, len(line_list)):
            sentence_list = line_list[i].split(" ")
            sentence_list_without_next_line = []
            for j in range(0, len(sentence_list)):
                word = sentence_list[j]
                if '\n' in word:
                    word2 = word.replace('\n', '')
                else:
                    word2 = word
                sentence_list_without_next_line.append(word2)
            all_whole_sentence.append(sentence_list_without_next_line)
        with open('data/temporaryData/'+'data'+str(FLAGS.year)+'_unique.txt', 'w') as test_data_unique:
            dic_words = {}
            for i in range(0, len(all_whole_sentence)):
                count += 1
                count2 += 1
                if count % 3 == 0 or count2 % 3 == 0:
                    for j in range(0, len(all_whole_sentence[i])):
                        if all_whole_sentence[i][j] == '$T$':
                            pass
                        elif not all_whole_sentence[i][j] in dic_words:
                            dic_words[all_whole_sentence[i][j]] = 0
                        else:
                            past_value = dic_words[all_whole_sentence[i][j]]
                            dic_words[all_whole_sentence[i][j]] = past_value + 1

                        if not all_whole_sentence[i][j] == '$T$':
                            all_whole_sentence[i][j] = all_whole_sentence[i][j] + '_' + str(dic_words[all_whole_sentence[i][j]])

                    sent = ' '.join(all_whole_sentence[i])
                    test_data_unique.write(sent + '\n')
                else:
                    test_data_unique.write(line_list[i])
    test_data1.close()
    test_data_unique.close()

    # split data in train data and test data
    with open('data/temporaryData/' + 'data'+str(FLAGS.year) +'_unique.txt', 'r') as test_data:
        line_list = test_data.readlines()
        with open('data/programGeneratedData/' + 'train_data' + str(FLAGS.year) + '_H=' + str(FLAGS.hops) + '.txt', 'w') as train_data:
            for i in range(0, number):
                train_data.write(line_list[i])
    test_data.close()
    train_data.close()

    with open('data/temporaryData/' + 'data' + str(FLAGS.year) + '_unique.txt', 'r') as td:
        line_list = td.readlines()
        with open('data/programGeneratedData/' + 'test_data' + str(FLAGS.year) + '_H=' + str(FLAGS.hops)+'.txt', 'w') as test_data:
            for i in range(number, len(line_list)):
                test_data.write(line_list[i])

    td.close()
    test_data.close()


