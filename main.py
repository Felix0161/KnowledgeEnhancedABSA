# https://github.com/mtrusca/HAABSA_PLUS_PLUS
# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from loadData import *

# import parameter configuration and data paths
from config import *

import lcrModelAlt_hierarchical_v4

# main function
def main(_):
    loadData = False  # only for non-contextualised word embeddings.
    weightanalysis = False
    runLCRROTALT_v4 = True


    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87

    # LCR-Rot-hop-ont++ model
    if runLCRROTALT_v4:
        _, pred1, fw1, bw1, tl1, tr1 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, FLAGS.test_path, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if weightanalysis:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path,
                                                                        FLAGS.test_path, accuracyOnt,
                                                                        test_size,
                                                                        remaining_size, emb_path=FLAGS.embedding_path_weight_analysis)
        tf.reset_default_graph()

        outF= open('sentence_analysis.txt', "w")
        dif = np.subtract(pred2, pred1)
        for i, value in enumerate(pred1):
            if value != pred2[i]:
                sentleft, sentright = [], []
                flag = True
                for word in sent[i]:
                    if word == '$t$':
                        flag = False
                        continue
                    if flag:
                        sentleft.append(word)
                    else:
                        sentright.append(word)
                print(i)
                outF.write(str(i))
                outF.write("\n")
                outF.write('original: {}; other: {}; true: {}'.format(pred1[i], pred2[i], true[i]))
                outF.write("\n")
                outF.write(";".join(sentleft))
                outF.write("\n")
                outF.write(";".join(str(x) for x in fw1[i][0]))
                outF.write("\n")
                outF.write(";".join(sentright))
                outF.write("\n")
                outF.write(";".join(str(x) for x in bw1[i][0]))
                outF.write("\n")
                outF.write(";".join(target[i]))
                outF.write("\n")
                outF.write(";".join(str(x) for x in tl1[i][0]))
                outF.write("\n")
                outF.write(";".join(str(x) for x in tr1[i][0]))
                outF.write("\n")
                outF.write(";".join(sentleft))
                outF.write("\n")
                outF.write(";".join(str(x) for x in fw2[i][0]))
                outF.write("\n")
                outF.write(";".join(sentright))
                outF.write("\n")
                outF.write(";".join(str(x) for x in bw2[i][0]))
                outF.write("\n")
                outF.write(";".join(target[i]))
                outF.write("\n")
                outF.write(";".join(str(x) for x in tl2[i][0]))
                outF.write("\n")
                outF.write(";".join(str(x) for x in tr2[i][0]))
                outF.write("\n")
        outF.close()

print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.app.run()