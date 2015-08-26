package org.maochen.nlp.parser.stanford.pcfg;

import com.google.common.collect.Table;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DTree;

import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

import edu.stanford.nlp.io.ExtensionFileFilter;
import edu.stanford.nlp.io.NumberRangeFileFilter;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.Options;
import edu.stanford.nlp.trees.DiskTreebank;
import edu.stanford.nlp.trees.Tree;

/**
 * Stanford PCFG Parser Trainer.
 *
 * <p> Created by Maochen on 1/9/15.
 */
public class StanfordPCFGParserTrainer {
    public static String wsj = null;
    public static String extra = null;
    public static String taggedFiles = null;
    public static String modelPath = null;

    private static void trainEngine(String trainDirPath, int startRange, int endRange, String train2DirPath, String train2FileExtension, double extraTrainingSetWeight, String modelPath, int maxLength, String taggedFiles) {
        File f = new File(modelPath);
        if (f.exists()) {
            System.out.println("Delete the existing model in " + f.getAbsolutePath());
            f.delete();
        }


        List<String> para = new ArrayList<>();
        para.add("-goodPCFG");
        para.add("-maxLength");
        para.add(String.valueOf(maxLength));
        para.add("-trainingThreads");
        para.add(String.valueOf(Runtime.getRuntime().availableProcessors()));
        para.add("-wordFunction");
        para.add("edu.stanford.nlp.process.AmericanizeFunction");

        if (taggedFiles != null) {
            para.add("-taggedFiles");
            para.add("tagSeparator=_," + taggedFiles);
        }

        Options op = new Options();
        op.setOptions(para.stream().toArray(String[]::new));

        DiskTreebank trainTreeBank = new DiskTreebank();
        FileFilter trainTreeBankFilter = new NumberRangeFileFilter(startRange, endRange, true);
        trainTreeBank.loadPath(trainDirPath, trainTreeBankFilter);

        DiskTreebank extraTreeBank = null;
        if (train2DirPath != null) {
            extraTreeBank = new DiskTreebank();
            FileFilter extraTreeBankFilter = new ExtensionFileFilter(train2FileExtension, true);
            extraTreeBank.loadPath(train2DirPath, extraTreeBankFilter);
        }

        LexicalizedParser.getParserFromTreebank(trainTreeBank, extraTreeBank, extraTrainingSetWeight, null, op, null, null).saveParserToSerialized(modelPath);
    }

    public static String train() throws IOException {
        trainEngine(wsj, 1, 2502, extra, ".mrg", 1.0, modelPath, 40, taggedFiles);
        return modelPath;
    }

    public static void main(String[] args) throws IOException {
        wsj = "/Users/Maochen/Desktop/treebank_3/parsed/mrg/wsj/";
        extra = "/Users/Maochen/Desktop/extra/treebank_extra_data/";
        taggedFiles = extra + "/train-tech-english";
        modelPath = "/Users/Maochen/workspace/nlpservice/nlp-service-remote/src/main/resources/classifierData/englishPCFG.ser.gz";

        train();

        StanfordPCFGParser parser = new StanfordPCFGParser(modelPath, null, null);
        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;
        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {

                Table<DTree, Tree, Double> result = parser.getKBestParse(input, 5);
                List<Table.Cell<DTree, Tree, Double>> list = result.cellSet().stream()
                        .sorted((c1, c2) -> c2.getValue().compareTo(c1.getValue()))
                        .collect(Collectors.toList());

                for (Table.Cell<DTree, Tree, Double> c : list) {
                    System.out.println(c.getValue());
                    System.out.println(c.getColumnKey().pennString());
                    System.out.println(c.getRowKey());
                    System.out.println("----------------------------");
                }
            }
        }
    }

}
