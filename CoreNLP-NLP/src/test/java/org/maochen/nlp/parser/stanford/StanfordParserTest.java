package org.maochen.nlp.parser.stanford;

import com.google.common.collect.Table;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;

import edu.stanford.nlp.trees.Tree;

/**
 * StanfordPCFGParser class no longer has dep to the default model. Do the main here.
 *
 * <p>Created by Maochen on 9/16/15.
 */
public class StanfordParserTest {


    public static void main(String[] args) {
        String modelFile = "/Users/Maochen/workspace/nlpservice/nlp-service-remote/src/main/resources/classifierData/englishPCFG.ser.gz";
        String posTaggerModel = null;//"/Users/Maochen/workspace/nlpservice/nlp-service-remote/src/main/resources/classifierData/english-left3words-distsim.tagger";
        StanfordPCFGParser parser = new StanfordPCFGParser(modelFile, posTaggerModel, new ArrayList<>());

        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;

        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
                // System.out.println(parser.parse(input).toString());
                Table<DTree, Tree, Double> trees = parser.getKBestParse(input, 3);

                List<Table.Cell<DTree, Tree, Double>> results = trees.cellSet().parallelStream().collect(Collectors.toList());
                results.sort((o1, o2) -> Double.compare(o2.getValue(), o1.getValue()));
                for (Table.Cell<DTree, Tree, Double> entry : results) {
                    System.out.println("--------------------------");
                    System.out.println(entry.getValue());
                    System.out.println(entry.getColumnKey().pennString());
                    System.out.println(StringUtils.EMPTY);
                    System.out.println(entry.getRowKey());
                }

            }
        }

    }

}
