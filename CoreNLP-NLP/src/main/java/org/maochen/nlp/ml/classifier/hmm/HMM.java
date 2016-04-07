package org.maochen.nlp.ml.classifier.hmm;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMM {
    private static final Logger LOG = LoggerFactory.getLogger(HMM.class);

    public static final int WORD_INDEX = 0;

    protected static final String START = "<START>";
    protected static final String END = "<END>";

    private static SequenceTuple getSequenceTuple(List<String> words, List<String> pos) {
        Map<Integer, List<String>> wordFeat = new HashMap<>();
        wordFeat.put(WORD_INDEX, words);
        return new SequenceTuple(wordFeat, pos);
    }

    public static List<SequenceTuple> readTrainFile(String filename, String delimiter, int wordColIndex, int tagColIndex) {
        List<SequenceTuple> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            List<String> words = new ArrayList<>();
            List<String> pos = new ArrayList<>();

            while (line != null) {
                if (line.trim().isEmpty()) {
                    if (!words.isEmpty() && !pos.isEmpty()) {
                        SequenceTuple tuple = getSequenceTuple(words, pos);
                        data.add(tuple);
                        words = new ArrayList<>();
                        pos = new ArrayList<>();
                    }
                } else {
                    String[] tp = line.split(delimiter);
                    String word = tp[wordColIndex];
                    words.add(WordUtils.normalizeWord(word));
                    pos.add(WordUtils.normalizeTag(tp[tagColIndex]));
                }
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }

    private static Consumer<Map<String, Double>> normalize = map -> {
        double total = map.values().stream().mapToDouble(x -> x).sum();

        for (String key : map.keySet()) {
            map.put(key, map.get(key) / total);
        }
    };

    static void normalizeEmission(HMMModel model) {
        model.emission.columnMap().values().stream().forEach(normalize);

        for (String tag : model.emission.columnKeySet()) { //use for oov pos tag.
            double minProb = model.emission.column(tag).values().stream().min(Double::compareTo).orElse(0.0);
            model.emissionMin.put(tag, minProb);
        }
    }

    static void normalizeTrans(HMMModel model) {
        model.transition.rowMap().values().stream().forEach(normalize);
    }

    private static Pair<List<String>, List<String>> getXSeqOSeq(SequenceTuple seqTuple) {
        List<String> words = seqTuple.entries.stream()
                .map(entry -> ((LabeledVector) entry.vector).featsName[WORD_INDEX])
                .collect(Collectors.toList());
        List<String> tag = seqTuple.getLabel();

        words.add(0, START);
        words.add(END);
        tag.add(0, START);
        tag.add(END);

        return new ImmutablePair<>(words, tag);
    }

    public static HMMModel train(List<SequenceTuple> data) {
        HMMModel model = new HMMModel();

        for (SequenceTuple seqTuple : data) {

            Pair<List<String>, List<String>> wordTagPair = getXSeqOSeq(seqTuple);
            List<String> words = wordTagPair.getLeft();
            List<String> tag = wordTagPair.getRight();

            for (int i = 0; i < words.size(); i++) {
                Double ct = model.emission.get(words.get(i), tag.get(i));  // Oi, Xi
                ct = ct == null ? 1 : ct + 1;
                model.emission.put(words.get(i), tag.get(i), ct);
            }

            for (int i = 0; i < seqTuple.entries.size() - 1; i++) {  // Xi -> X_i+1
                Double ct = model.transition.get(tag.get(i), tag.get(i + 1));
                ct = ct == null ? 1 : ct + 1;
                model.transition.put(tag.get(i), tag.get(i + 1), ct);
            }
        }

        normalizeEmission(model);
        normalizeTrans(model);
        return model;
    }

    public static List<String> viterbi(HMMModel model, String[] words) {
        return Viterbi.resolve(model, words);
    }

    public static void eval(HMMModel model, String testFile, String delimiter, int wordColIndex, int tagColIndex) {
        List<SequenceTuple> testData = readTrainFile(testFile, delimiter, wordColIndex, tagColIndex);
        int totalCount = 0;
        int errCount = 0;

        for (SequenceTuple sequenceTuple : testData) {
            String[] words = sequenceTuple.entries.stream().map(entry -> ((LabeledVector) entry.vector).featsName[WORD_INDEX]).toArray(String[]::new);
            List<String> result = viterbi(model, words);

            for (int i = 0; i < result.size(); i++) {
                totalCount++;
                String expected = WordUtils.normalizeTag(sequenceTuple.entries.get(i).label);
                String actual = WordUtils.normalizeTag(result.get(i));
                if (!(actual.startsWith(expected) || expected.startsWith(actual))) {
                    System.out.println(words[i] + " exp: " + expected + " actual: " + result.get(i));
                    errCount++;
                }
            }

        }

        double accurancy = (1 - errCount / (double) totalCount) * 100;
        System.out.println("accurancy: " + errCount + "/" + totalCount + " -> " + String.format("%.2f", accurancy) + "%");
    }

    public static HMMModel loadModel(String modelPath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath))) {
            return (HMMModel) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            LOG.error("Load model err.", e);
        }
        return null;
    }

    public static void saveModel(String modelPath, HMMModel model) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath))) {
            oos.writeObject(model);
        } catch (IOException e) {
            LOG.error("Persist model err.", e);
        }
    }

    public static void main(String[] args) throws InterruptedException {
//        Thread.sleep(5000);

        String prefix = "/Users/mguan/Dropbox/Course/Natural Lang Processing/HW/HW4_POSTagger_HMM/Homework4_corpus/POSData";
        List<SequenceTuple> data = HMM.readTrainFile(prefix + "/development.pos", "\t", 0, 1);
        List<SequenceTuple> data2 = HMM.readTrainFile(prefix + "/training.pos", "\t", 0, 1);
        data.addAll(data2);

        HMMModel model = train(data);
        eval(model, prefix + "/training.pos", "\t", 0, 1);

        // Please add dot in the end. All training data ends with dot, so the transition from anything other than dot to <END> is 0
        String str = "The quick brown fox jumped over the lazy dog .";
        List<String> result = viterbi(model, str.split("\\s"));
        System.out.println(str);
        System.out.println(result);
    }
}
