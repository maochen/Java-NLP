package org.maochen.parser.stanford.nn;

import edu.stanford.nlp.io.ExtensionFileFilter;
import edu.stanford.nlp.trees.*;
import org.maochen.parser.StanfordParserUtils;

import java.io.FileFilter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Collection;

/**
 * WIP
 * Created by Maochen on 4/13/15.
 */
public class StanfordNNParserTrainer {

    public static final String conllXTrainFile = "/Users/Maochen/Desktop/train.conllx.txt";
    public static final String modelPath = "/Users/Maochen/Desktop/nndep.ser.gz";

    public static final String WSJ = "/Users/Maochen/Desktop/treebank_3/parsed/mrg/wsj";
    public static final String extra = "/Users/Maochen/Desktop/extra/treebank_extra_data/";

    private static void count(int counter, Treebank trainTreeBank) {
        counter++;
        if (counter % 1000 == 0) {
            System.out.println("Processing " + counter + " of " + trainTreeBank.size());
        }
    }

    public static void convertTreebankToCoNLLX(String trainDirPath, FileFilter trainTreeBankFilter, boolean makeCopulaVerbHead, String outputFileName) {
        int counter = 0;
        DiskTreebank trainTreeBank = new DiskTreebank();
        trainTreeBank.loadPath(trainDirPath, trainTreeBankFilter);

        SemanticHeadFinder headFinder = new SemanticHeadFinder(!makeCopulaVerbHead); // keep copula verbs as head

        try {
            FileWriter fw = new FileWriter(outputFileName);

            trainTreeBank.parallelStream().forEach(tree -> {
                count(counter, trainTreeBank);
                Collection<TypedDependency> tdep = new EnglishGrammaticalStructure(tree, string -> true, headFinder, true).typedDependencies();
                String conllxString = StanfordParserUtils.getCoNLLXString(tdep, tree.taggedLabeledYield());
                try {
                    fw.write(conllxString);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            fw.flush();
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void train() {

        FileFilter filter = new ExtensionFileFilter(".mrg", true);
        // FileFilter trainTreeBankFilter = new NumberRangeFileFilter(1, 2502, true);

        // 1,2502
        convertTreebankToCoNLLX(WSJ, filter, true, conllXTrainFile);
        convertTreebankToCoNLLX(extra, filter, true, "/Users/Maochen/Desktop/extra.conllx.txt");

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
