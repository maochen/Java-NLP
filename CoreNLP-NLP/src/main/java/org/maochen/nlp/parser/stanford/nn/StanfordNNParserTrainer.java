package org.maochen.nlp.parser.stanford.nn;

import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Treebank;
import edu.stanford.nlp.trees.UniversalPOSMapper;

/**
 * WIP
 * Created by Maochen on 4/13/15.
 */
public class StanfordNNParserTrainer {

    public static final String conllXTrainFile = "/Users/Maochen/Desktop/train.conllx.txt";
    public static final String modelPath = "/Users/Maochen/Desktop/nndep.ser.gz";

    public static final String WSJ = "/Users/Maochen/Desktop/treebank_3/parsed/mrg/wsj";
    public static final String extra = "/Users/Maochen/Desktop/extra/treebank_extra_data/";

    public static void train(final String conllXTrainFile, final String outputFilePath) {
        StanfordNNDepParser stanfordNNDepParser = new StanfordNNDepParser();
//        stanfordNNDepParser.nndepParser.
    }

    public static void getCPosTag() {
        Treebank treeBank = new DiskTreebank();
        treeBank.loadPath("/Users/Maochen/Desktop/temp.txt");
        Tree tree = treeBank.iterator().next();

        System.out.println(tree.pennString());
        Tree uposTree = UniversalPOSMapper.mapTree(tree);

        System.out.println(uposTree.pennString());
    }
}
