package org.maochen.parser.stanford.pcfg;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import org.apache.commons.lang3.StringUtils;
import org.maochen.datastructure.DTree;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Give a bunch of sentences, generate parse trees.
 *
 * Created by Maochen on 5/26/15.
 */
public class TrainingDataGenerator {
    private static final String modelFile = "/Users/Maochen/workspace/nlpservice/nlp-service-remote/src/main/resources/classifierData/englishPCFG.ser.gz";
    private static final String posTaggerModel = "/Users/Maochen/workspace/nlpservice/nlp-service-remote/src/main/resources/classifierData/english-left3words-distsim.tagger";
    private static final StanfordPCFGParser parser = new StanfordPCFGParser(modelFile, posTaggerModel, true);

    private static List<String> splitSentences(String input) {
        DocumentPreprocessor dp = new DocumentPreprocessor(new StringReader(input));
        Function<List<HasWord>, String> stringFunction = hasWords -> hasWords.stream().map(HasWord::word)
                .reduce((x, y) -> x + StringUtils.SPACE + y)
                .get();

        List<String> sentenceList = StreamSupport.stream(dp.spliterator(), false)
                .map(stringFunction).collect(Collectors.toList());


        for (int i = 0; i < sentenceList.size(); i++) {
            String sentence = sentenceList.get(i);
            sentence = sentence.replaceAll("-LSB-", "[");
            sentence = sentence.replaceAll("-RSB-", "]");
            sentence = sentence.replaceAll("-LRB-", "(");
            sentence = sentence.replaceAll("-RRB-", ")");
            sentence = sentence.replaceAll("-LCB-", "{");
            sentence = sentence.replaceAll("-RCB-", "}");
            sentence = sentence.replaceAll("\\s+((\\,|\\.|\\!|\\?)+)", "$1");
            sentence = sentence.replaceAll("(\\$)\\s+(\\d+)", "$1$2"); // Strip off $ 1234 -> $1234
            sentence = sentence.replaceAll("(\\w+)\\s+('|\")", "$1$2");
            sentence = sentence.replaceAll("(\\(|`)\\s+(\\w+)", "$1$2"); // Strip off ( ss -> (ss, ' hello -> 'hello
            sentence = sentence.replaceAll("``", "\"");
            sentence = sentence.replaceAll("`", "'");
            sentence = sentence.replaceAll("(\\w+)\\s+(\\)|'')", "$1$2"); // Strip off ss ) -> ss)
            sentence = sentence.replaceAll("''", "\"");
            sentence = sentence.replaceAll("(\\w+)\\s+:", "$1:"); // current user : -> current user:
            sentence = sentence.replaceAll("(\\w+)\\s+'s", "$1's"); // He 's only five

            sentenceList.remove(i);
            sentenceList.add(i, sentence);
        }


        return sentenceList;
    }

    public void generate(String inputFilename, String outputFileName) {
        List<String> sentences = readFile(inputFilename);

        int count = 0;
        try {
            BufferedWriter output = new BufferedWriter(new FileWriter(new File(outputFileName)));
            for (String sentenceUnsegmented : sentences) {
                count++;
                if (count % 20 == 0) {
                    System.out.println("Processed:\t" + count);
                }
                List<String> sentenceSegmented = splitSentences(sentenceUnsegmented);
                for (String sentence : sentenceSegmented) {
                    DTree tree = parser.parse(sentence);
                    output.write(tree.toString() + System.lineSeparator());
                }
            }

            output.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static List<String> readFile(final String fileName) {
        List<String> results = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line = br.readLine();

            while (line != null) {
                results.add(line.trim());
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return results;
    }

    public static void main(String[] args) {
        TrainingDataGenerator tdg = new TrainingDataGenerator();
        tdg.generate("/Users/Maochen/Desktop/clean.txt", "/Users/Maochen/Desktop/out.txt");
    }
}
