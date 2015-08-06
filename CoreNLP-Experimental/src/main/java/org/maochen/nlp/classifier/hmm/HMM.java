package org.maochen.nlp.classifier.hmm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMM {
    protected static final String START = "<START>";
    protected static final String END = "<END>";

    public static List<HMMTuple> readFile(String filename, String delimiter, int wordColIndex, int tagColIndex) {
        List<HMMTuple> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line = br.readLine();
            List<String> words = new ArrayList<>();
            List<String> pos = new ArrayList<>();

            while (line != null) {
                if (line.trim().isEmpty()) {
                    if (!words.isEmpty() && !pos.isEmpty()) {
                        HMMTuple tuple = new HMMTuple(words, pos);
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

    public static void normalizeEmission(HMMModel model) {
        model.emission.columnMap().values().parallelStream().map(cols -> {
                    double total = cols.values().stream().mapToDouble(x -> x).sum();
                    for (String row : cols.keySet()) {
                        cols.put(row, cols.get(row) / total);
                    }

                    return null;
                }
        ).collect(Collectors.toSet());

        for (String tag : model.emission.columnKeySet()) { //use for oov pos tag.
            double minProb = model.emission.column(tag).values().stream().min(Double::compareTo).orElse(0.0);
            model.emissionMin.put(tag, minProb);
        }
    }

    public static void normalizeTrans(HMMModel model) {
        model.transition.rowMap().values().parallelStream().map(rows -> {
                    double total = rows.values().stream().mapToDouble(x -> x).sum();
                    for (String col : rows.keySet()) {
                        rows.put(col, rows.get(col) / total);
                    }

                    return null;
                }
        ).collect(Collectors.toSet());
    }

    public static HMMModel train(List<HMMTuple> data) {
        HMMModel model = new HMMModel();

        for (HMMTuple tuple : data) {
            tuple.words.add(0, START);
            tuple.words.add(END);
            tuple.tag.add(0, START);
            tuple.tag.add(END);

            for (int i = 0; i < tuple.words.size(); i++) {
                Double ct = model.emission.get(tuple.words.get(i), tuple.tag.get(i));
                ct = ct == null ? 1 : ct + 1;
                model.emission.put(tuple.words.get(i), tuple.tag.get(i), ct);
            }

            for (int i = 0; i < tuple.tag.size() - 1; i++) {
                Double ct = model.transition.get(tuple.tag.get(i), tuple.tag.get(i + 1));
                ct = ct == null ? 1 : ct + 1;
                model.transition.put(tuple.tag.get(i), tuple.tag.get(i + 1), ct);
            }
        }

        normalizeEmission(model);
        normalizeTrans(model);
        return model;
    }


    public static List<String> viterbi(HMMModel model, List<String> words) {
        return Viterbi.resolve(model, words);
    }

    public static void eval(HMMModel model, String testFile, String delimiter, int wordColIndex, int tagColIndex) {
        List<HMMTuple> testData = readFile(testFile, delimiter, wordColIndex, tagColIndex);
        int totalcount = 0;
        int errcount = 0;

        for (HMMTuple tuple : testData) {
            List<String> result = viterbi(model, tuple.words);

            for (int i = 0; i < result.size(); i++) {
                totalcount++;
                String expected = WordUtils.normalizeTag(tuple.tag.get(i));
                String actual = WordUtils.normalizeTag(result.get(i));
                if (!(actual.startsWith(expected) || expected.startsWith(actual))) {
                    System.out.println(tuple.words.get(i) + " exp: " + expected + " actual: " + result.get(i));
                    errcount++;
                }
            }

        }

        double accurancy = (1 - errcount / (double) totalcount) * 100;
        System.out.println("accurancy: " + errcount + "/" + totalcount + " -> " + String.format("%.2f", accurancy) + "%");
    }

    public static void main(String[] args) throws InterruptedException {
//        Thread.sleep(5000);

        List<HMMTuple> data = HMM.readFile("/Users/Maochen/Desktop/POS/penntreebank.txt", "\t", 1, 4);
        List<HMMTuple> data2 = HMM.readFile("/Users/Maochen/Desktop/POS/extra.txt", "\t", 1, 4);
        List<HMMTuple> data3 = HMM.readFile("/Users/Maochen/Desktop/POS/training.pos", "\t", 0, 1);

        data.addAll(data2);
        data.addAll(data3);

        HMMModel model = train(data);
//        eval(model, "/Users/Maochen/Desktop/POS/test.pos", "\t", 0, 1);

        String str = "The quick brown fox jumped over the lazy dog";
        List<String> result = viterbi(model, Arrays.asList(str.split("\\s")));
        System.out.println(result);
    }
}
