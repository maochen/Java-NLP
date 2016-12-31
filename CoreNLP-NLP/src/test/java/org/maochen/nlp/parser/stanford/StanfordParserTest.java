package org.maochen.nlp.parser.stanford;

import static junit.framework.Assert.assertEquals;

import com.google.common.collect.Table;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.trees.Tree;
import org.apache.commons.lang3.StringUtils;
import org.junit.Test;
import org.maochen.nlp.parser.DTree;
import org.maochen.nlp.parser.IParser;
import org.maochen.nlp.parser.stanford.nn.StanfordNNDepParser;
import org.maochen.nlp.parser.stanford.pcfg.StanfordPCFGParser;
import org.maochen.nlp.parser.stanford.util.StanfordConst;

import java.util.List;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * StanfordPCFGParser class no longer has dep to the default model. Do the main here.
 * <p>
 * <p>Created by Maochen on 9/16/15.
 */
public class StanfordParserTest {

  private static final IParser PARSER = new StanfordPCFGParser(null, null, null);

  @Test
  public void testPositionStartEnd() {
    String sentence = "This is Tom's cat (A).";
    DTree tree = PARSER.parse(sentence);

    int[] expectedStart = new int[]{0, 5, 8, 11, 14, 18, 19, 20, 21};
    int[] expectedEnd = new int[]{4, 7, 11, 13, 17, 19, 20, 21, 22};

    IntStream.range(1, tree.size()).forEach(i -> {
      int actualStart = Integer.valueOf(tree.get(i).getFeature("index_start"));
      int actualEnd = Integer.valueOf(tree.get(i).getFeature("index_end"));

      assertEquals(expectedStart[i - 1], actualStart);
      assertEquals(expectedEnd[i - 1], actualEnd);
    });
  }

  @Test
  public void testSentenceRecoverIveacar() {
    String sentence = "I've a car.";
    DTree tree = PARSER.parse(sentence);
    assertEquals(sentence, tree.sentence());
  }

  @Test
  public void testSentenceRecoverQuotes() {
    String sentence = "I say : \" This is a guy.\"";
    DTree tree = PARSER.parse(sentence);
    assertEquals(sentence, tree.sentence());
  }

  public static void main(String[] args) {
    boolean usePCFG = true;
    StanfordPCFGParser pcfgParser = null;
    IParser nnParser = null;

    if (usePCFG) {
      pcfgParser = new StanfordPCFGParser(StanfordConst.STANFORD_DEFAULT_PCFG_EN_MODEL, null, null);
    } else {
      nnParser = new StanfordNNDepParser(DependencyParser.DEFAULT_MODEL, null, null);
    }

    Scanner scan = new Scanner(System.in);
    String input = StringUtils.EMPTY;

    String quitRegex = "q|quit|exit";
    while (!input.matches(quitRegex)) {
      System.out.println("Please enter sentence:");
      input = scan.nextLine();
      if (!input.trim().isEmpty() && !input.matches(quitRegex)) {

        if (usePCFG) {
          // System.out.println(parser.parse(input).toString());
          Table<DTree, Tree, Double> trees = pcfgParser.getKBestParse(input, 3);

          List<Table.Cell<DTree, Tree, Double>> results = trees.cellSet().parallelStream().collect(Collectors.toList());
          results.sort((o1, o2) -> Double.compare(o2.getValue(), o1.getValue()));
          for (Table.Cell<DTree, Tree, Double> entry : results) {
            System.out.println("--------------------------");
            System.out.println(entry.getValue());
            System.out.println(entry.getColumnKey().pennString());
            System.out.println(StringUtils.EMPTY);
            System.out.println("Sentence: " + entry.getRowKey().sentence());
            System.out.println(StringUtils.EMPTY);
            System.out.println(entry.getRowKey());
          }
        } else {
          DTree tree = nnParser.parse(input);
          System.out.println(tree);
        }
      }
    }
  }
}
