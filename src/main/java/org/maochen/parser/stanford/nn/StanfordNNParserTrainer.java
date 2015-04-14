package org.maochen.parser.stanford.nn;

import edu.stanford.nlp.io.NumberRangeFileFilter;
import edu.stanford.nlp.trees.*;
import org.maochen.parser.StanfordParserUtils;

import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;

/**
 * Created by Maochen on 4/13/15.
 */
public class StanfordNNParserTrainer {

    public static final String conllXTrainFile = "/Users/Maochen/Desktop/train.conllx.txt";
    public static final String modelPath = "/Users/Maochen/Desktop/nndep.ser.gz";

    public static final String WSJ = "/Users/Maochen/Desktop/treebank_3/parsed/mrg/wsj";

    public static void convertTreebankToCoNLLX(String trainDirPath, int startRange, int endRange, boolean makeCopulaVerbHead) {
        DiskTreebank trainTreeBank = new DiskTreebank();
        FileFilter trainTreeBankFilter = new NumberRangeFileFilter(startRange, endRange, true);
        trainTreeBank.loadPath(trainDirPath, trainTreeBankFilter);

        SemanticHeadFinder headFinder = new SemanticHeadFinder(!makeCopulaVerbHead); // keep copula verbs as head

        try {
            FileWriter fw = new FileWriter(conllXTrainFile);

            for (Tree tree : trainTreeBank) {
                Collection<TypedDependency> tdep = new EnglishGrammaticalStructure(tree, string -> true, headFinder, true).typedDependencies();
                String conllxString = StanfordParserUtils.getCoNLLXString(tdep, tree.taggedLabeledYield());
                fw.write(conllxString);
            }

            fw.flush();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void train() {
        convertTreebankToCoNLLX(WSJ, 1, 2, true);
        //        List<String> a = Lists.newArrayList("-treeFile", "/Users/Maochen/Desktop/tree.txt", "−basic", "−keepPunct", "−conllx");

    }

    public static void main(String[] args) {
        train();
    }

}
