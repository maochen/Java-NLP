package org.maochen.nlp.parser.stanford.nn;

import org.maochen.nlp.parser.StanfordParserUtils;

import java.io.FileFilter;

/**
 * Created by Maochen on 4/13/15.
 */
public class StanfordNNParserTrainer {

    public static void getCoNLLXFromPennTreebank(FileFilter filter, String pennTreeFolder, String outputFile) {
        StanfordParserUtils.convertTreebankToCoNLLX(pennTreeFolder, filter, outputFile);
    }

    public static void train(String conllTrainFile, String wordEmbeddingFile, String outputModelPath, String preModel) {
        StanfordNNDepParser nnDepParser = new StanfordNNDepParser();
        nnDepParser.nndepParser.train(conllTrainFile, null, outputModelPath, wordEmbeddingFile, preModel);
    }
//    public static void getCPosTag(Tree tree) {
//        Treebank treeBank = new DiskTreebank();
//        treeBank.loadPath("/Users/Maochen/Desktop/temp.txt");
//        Tree tree = treeBank.iterator().next();
//
//        System.out.println(tree.pennString());
//        Tree uposTree = UniversalPOSMapper.mapTree(tree);
//
//        System.out.println(uposTree.pennString());
//    }
}
