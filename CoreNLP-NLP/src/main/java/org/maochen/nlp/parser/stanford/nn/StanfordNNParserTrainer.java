package org.maochen.nlp.parser.stanford.nn;

/**
 * Created by Maochen on 4/13/15.
 */
public class StanfordNNParserTrainer {
    public static void train(String conllTrainFile, String wordEmbeddingFile, String outputModelPath, String preModel) {
        StanfordNNDepParser nnDepParser = new StanfordNNDepParser();
        nnDepParser.nndepParser.train(conllTrainFile, null, outputModelPath, wordEmbeddingFile, preModel);
    }
}
