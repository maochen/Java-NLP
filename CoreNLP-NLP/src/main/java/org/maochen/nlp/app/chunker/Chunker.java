package org.maochen.nlp.app.chunker;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.app.ISeqTagger;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.maochen.nlp.parser.stanford.StanfordParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

/**
 * Created by Maochen on 11/10/15.
 */
public class Chunker extends MaxEntClassifier implements ISeqTagger {

    private static final Logger LOG = LoggerFactory.getLogger(Chunker.class);

    public static final int WORD_INDEX = 0;
    public static final int POS_INDEX = 1;

    private static MaxentTagger POS_TAGGER = null;

    public static String TRAIN_FILE_DELIMITER = "\t";

    private static Set<SequenceTuple> readFile(String fileName) {
        Set<SequenceTuple> trainingData = new HashSet<>();

        List<String> words = new ArrayList<>();
        List<String> pos = new ArrayList<>();
        List<String> tag = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line = br.readLine();
            while (line != null) {
                if (!line.trim().isEmpty()) {
                    String[] fields = line.split(TRAIN_FILE_DELIMITER);
                    words.add(fields[0]);
                    pos.add(fields[1]);
                    tag.add(fields[2]); // chunker label.
                } else {
                    Map<Integer, List<String>> feats = new HashMap<>();
                    feats.put(WORD_INDEX, words);
                    feats.put(POS_INDEX, pos);
                    SequenceTuple current = new SequenceTuple(feats, tag);
                    trainingData.add(current);
                    words = new ArrayList<>();
                    pos = new ArrayList<>();
                    tag = new ArrayList<>();
                }

                line = br.readLine();
            }
        } catch (IOException e) {
            LOG.error("load data err.", e);
        }

        return trainingData;
    }

    @Override
    public void train(String trainFilePath) {
        // Preparing training data
        Set<SequenceTuple> trainingData = readFile(trainFilePath);
        // -----------
        LOG.info("Loaded Training data.");

        LOG.info("Generating feats");
        List<Tuple> trainingTuples = ChunkerFeatureExtractor.extract(trainingData);
        LOG.info("Extracted Feats.");
        super.train(trainingTuples);

    }

    @Override
    public SequenceTuple predict(String sentence) {
        List<CoreLabel> tokenizedSentence = StanfordParser.stanfordTokenize(sentence);
        if (POS_TAGGER == null) {
            String posTaggerModelPath = "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger";
            POS_TAGGER = new MaxentTagger(posTaggerModelPath);
        }

        List<String> words = tokenizedSentence.stream().map(CoreLabel::originalText).collect(Collectors.toList());
        List<String> pos = POS_TAGGER.tagSentence(tokenizedSentence).stream().map(TaggedWord::tag).collect(Collectors.toList());

        Map<Integer, List<String>> feats = new HashMap<>();
        feats.put(WORD_INDEX, words);
        feats.put(POS_INDEX, pos);

        List<String> tags = IntStream.range(0, feats.values().stream().findFirst().get().size())
                .mapToObj(i -> StringUtils.EMPTY)
                .collect(Collectors.toList());
        SequenceTuple current = new SequenceTuple(feats, tags);

        predict(current);

        return current;
    }

    public void validate(String testFile) {
        Set<SequenceTuple> testData = readFile(testFile);
        int errCount = 0;
        int total = 0;

        for (SequenceTuple st : testData) {
            total += st.tag.size();
            List<String> expectedTags = new ArrayList<>(st.tag);
            predict(st);

            boolean isThisSeqPrinted = false;
            for (int i = 0; i < expectedTags.size(); i++) {
                if (!expectedTags.get(i).equals(st.tag.get(i))) {
                    if (!isThisSeqPrinted) {
                        printSequenceTuple(st, expectedTags);
                        System.out.println("");
                        isThisSeqPrinted = true;
                    }
                    errCount++;
                }
            }
        }

        System.out.println("Err/Total:\t" + errCount + "/" + total);
        System.out.println("Accurancy:\t" + (1 - (errCount / (double) total)) * 100 + "%");
    }

    @Override
    public void predict(final SequenceTuple sequenceTuple) {
        if (sequenceTuple == null) {
            return;
        }

        String[] tokens = sequenceTuple.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[Chunker.WORD_INDEX]).toArray(String[]::new);
        String[] pos = sequenceTuple.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[Chunker.POS_INDEX]).toArray(String[]::new);

        for (int i = 0; i < sequenceTuple.entries.size(); i++) {

            String[] resolvedTags = new String[tokens.length];
            int size = Math.min(resolvedTags.length, sequenceTuple.tag.size());

            for (int j = 0; j < size; j++) {
                resolvedTags[j] = sequenceTuple.tag.get(j);
            }

            String[] currentFeats = ChunkerFeatureExtractor.extractFeatSingle(i, tokens, pos, resolvedTags).stream().toArray(String[]::new);

            Tuple t = new Tuple(new LabeledVector(currentFeats));

            t.label = super.predict(t).entrySet().stream()
                    .max((t1, t2) -> t1.getValue().compareTo(t2.getValue())).map(Map.Entry::getKey).get();

            sequenceTuple.tag.set(i, t.label);
        }
    }


    public static void printSequenceTuple(final SequenceTuple st, List<String> correctTag) {
        String[] tokens = st.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[Chunker.WORD_INDEX]).toArray(String[]::new);

        for (int i = 0; i < tokens.length; i++) {
            String out = tokens[i] + "\t" + st.tag.get(i);
            if (correctTag != null && !st.tag.get(i).equals(correctTag.get(i))) {
                out += "\t" + "Expected:\t" + correctTag.get(i);
            }
            System.out.println(out);
        }
    }

    public static void main(String[] args) throws IOException {
        Chunker chunker = new Chunker();
        Chunker.TRAIN_FILE_DELIMITER = StringUtils.SPACE;
        String modelPath = "/Users/mguan/Desktop/chunker.model";

        Properties para = new Properties();
        para.put("iterations", "1000");
        chunker.setParameter(para);

//        String trainFile = "/Users/mguan/Desktop/all.txt";
        String trainFile = "/Users/mguan/Desktop/CoNLL_2000_Chunking/train.txt";

        chunker.train(trainFile);
        chunker.persistModel(modelPath);

//        chunker.loadModel(new FileInputStream(new File(modelPath)));
        chunker.validate("/Users/mguan/Desktop/CoNLL_2000_Chunking/test.txt");

        Scanner scan = new Scanner(System.in);
        String input = StringUtils.EMPTY;

        String quitRegex = "q|quit|exit";
        while (!input.matches(quitRegex)) {
            System.out.println("Please enter sentence:");
            input = scan.nextLine();
            if (!input.trim().isEmpty() && !input.matches(quitRegex)) {
                SequenceTuple st = chunker.predict(input);
                printSequenceTuple(st, null);
            }
        }


    }
}
