package org.maochen.nlp.parser.stanford.nn;

import edu.stanford.nlp.io.ExtensionFileFilter;
import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Treebank;
import edu.stanford.nlp.trees.UniversalPOSMapper;
import org.maochen.nlp.parser.StanfordParserUtils;

import java.io.FileFilter;

/**
 * WIP
 * Created by Maochen on 4/13/15.
 */
public class StanfordNNParserTrainer {

    public static final String conllXTrainFile = "/Users/Maochen/Desktop/train.conllx.txt";
    public static final String modelPath = "/Users/Maochen/Desktop/nndep.ser.gz";

    public static final String WSJ = "/Users/Maochen/Desktop/treebank_3/parsed/mrg/wsj";
    public static final String extra = "/Users/Maochen/Desktop/extra/treebank_extra_data/";

    public static void train() {

        FileFilter filter = new ExtensionFileFilter(".mrg", true);
        // FileFilter trainTreeBankFilter = new NumberRangeFileFilter(1, 2502, true);

        // 1,2502
        //        StanfordParserUtils.convertTreebankToCoNLLX(WSJ, filter, true, conllXTrainFile);
        //        StanfordParserUtils.convertTreebankToCoNLLX(extra, filter, true, "/Users/Maochen/Desktop/extra.conllx.txt");
        StanfordParserUtils.convertTreebankToCoNLLX("/Users/Maochen/Desktop/extra/treebank_extra_data/maochen_hand_parsed/wsj_2502.mrg", filter, conllXTrainFile);
        // List<String> a = Lists.newArrayList("-treeFile", "/Users/Maochen/Desktop/tree.txt", "−basic", "−keepPunct", "−conllx");

    }

    public static void getCPosTag() {
        Treebank treeBank = new DiskTreebank();
        treeBank.loadPath("/Users/Maochen/Desktop/temp.txt");
        Tree tree = treeBank.iterator().next();

        System.out.println(tree.pennString());
        Tree uposTree = UniversalPOSMapper.mapTree(tree);

        System.out.println(uposTree.pennString());
    }

    public static void main(String[] args) {
        train();
    }

}
