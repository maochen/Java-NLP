package org.maochen.nlp.ml.util;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.maxent.MaxEntClassifier;
import org.maochen.nlp.ml.util.dataio.CSVDataReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/11/15.
 */
public class CrossValidation {
    private static final Logger LOG = LoggerFactory.getLogger(CrossValidation.class);

    public static class Score {
        int nfold;
        String label;
        int tp = 0;
        int tn = 0;
        int fp = 0;
        int fn = 0;

        public double getF1() {
            double precision = getPrecision();
            double recall = getRecall();
            return 2 * precision * recall / (precision + recall);
        }

        public double getF2() {
            double precision = getPrecision();
            double recall = getRecall();
            return 5 * precision * recall / (4 * precision + recall);
        }

        public double getPrecision() {
            return tp / (double) (tp + fp);
        }

        public double getRecall() {
            return tp / (double) (tp + fn);
        }

        public double getAccuracy() {
            return (tn + tp) / (double) (tp + tn + fn + fp);
        }

        @Override
        public String toString() {
            return "P: " + String.format("%.2f", getPrecision())
                    + "\tR: " + String.format("%.2f", getRecall())
                    + "\tA: " + String.format("%.2f", getAccuracy())
                    + "\tF1: " + String.format("%.2f", getF1());
        }
    }

    private int nfold;

    private IClassifier classifier;

    private Set<String> labels;

    // R, C, V -- nfold, label, score
    public Table<Integer, String, Score> scores = HashBasedTable.create();

    private boolean shuffleData;

    /**
     * Cross validation.
     *
     * @param data whole testing data collection
     */
    public void run(final List<Tuple> data) {
        List<Tuple> dataCopy = new ArrayList<>(data);
        this.labels = data.parallelStream().map(x -> x.label).collect(Collectors.toSet());
        if (shuffleData) {
            Collections.shuffle(dataCopy);
        }

        int chunkSize = data.size() / nfold;

        int reminder = data.size() % chunkSize;

        for (int i = data.size() - 1; i > data.size() - 1 - reminder; i--) {
            LOG.info("Dropping the tail id: " + data.get(i).id);
        }

        for (int i = 0; i < nfold; i++) {
            System.err.println("Cross validation round " + (i + 1) + "/" + nfold);
            List<Tuple> testing = new ArrayList<>(data.subList(i, i + chunkSize));
            List<Tuple> training = new ArrayList<>(data.subList(0, i));
            training.addAll(data.subList(i + chunkSize, data.size()));

            eval(training, testing, i);
        }
    }

    // This is for one fold.
    private void eval(List<Tuple> training, List<Tuple> testing, int nfold) {
        classifier.train(training);

        for (Tuple tuple : testing) {
            String actual = classifier.predict(tuple).entrySet().stream()
                    .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
                    .map(Map.Entry::getKey).orElse(StringUtils.EMPTY);
            updateScore(tuple, actual, nfold);
        }
    }

    private void updateScore(Tuple testingTuple, String actual, int nfold) {
        labels.stream()
                .filter(label -> !scores.contains(nfold, label))
                .forEach(label -> { // Init score.
                    Score score = new Score();
                    score.nfold = nfold;
                    score.label = label;
                    scores.put(nfold, label, score);
                });

        // Label, Score
        Map<String, Score> nfoldResult = scores.row(nfold);

        if (testingTuple.label.equals(actual)) { // Correct Predicted
            Score score = nfoldResult.get(testingTuple.label);
            score.tp++;

            // update all others.
            nfoldResult.entrySet().stream()
                    .filter(x -> !x.getKey().equals(testingTuple.label))
                    .map(Map.Entry::getValue)
                    .forEach(s -> s.tn++);

        } else { // Wrong predicted
            String wrongLabel = actual;
            String correctLabel = testingTuple.label;

            nfoldResult.get(wrongLabel).fp++;

            nfoldResult.entrySet().stream()
                    .filter(x -> !x.getKey().equals(correctLabel))
                    .map(Map.Entry::getValue)
                    .forEach(score -> score.fn++);

            //Rest
            nfoldResult.entrySet().stream()
                    .filter(x -> !x.getKey().equals(correctLabel) && !x.getKey().equals(wrongLabel))
                    .map(Map.Entry::getValue)
                    .forEach(score -> score.tn++);
        }
    }


    public Score getResult() {
        Score result = new Score();
        scores.values().stream().forEach(s -> {
            result.fn += s.fn;
            result.fp += s.fp;
            result.tn += s.tn;
            result.tp += s.tp;
        });

        result.fn /= scores.size();
        result.fp /= scores.size();
        result.tn /= scores.size();
        result.tp /= scores.size();
        return result;
    }

    /**
     * Constructor
     *
     * @param nfold       nfold for the cross validation. (Recommend: 10-fold)
     * @param classifier  the actual classifier need to test.
     * @param shuffleData whether the data needs to be shuffled at the begining of the whole
     *                    process.
     */
    public CrossValidation(final int nfold, final IClassifier classifier, final boolean shuffleData) {
        if (nfold < 2) {
            throw new RuntimeException("CV expects n-fold greater than 1.");
        }
        this.nfold = nfold;
        this.classifier = classifier;
        this.shuffleData = shuffleData;
    }


    public static void main(String[] args) throws IOException {
        IClassifier maxEntClassifier = new MaxEntClassifier();
        Properties properties = new Properties();
        properties.put("iter", "500");
        maxEntClassifier.setParameter(properties);
        String fileName = "/Users/mguan/Desktop/train.balanced.csv";
        CSVDataReader dataReader = new CSVDataReader(fileName, -1, ",", false, true);
        List<Tuple> data = dataReader.read();
        CrossValidation cv = new CrossValidation(10, maxEntClassifier, true);

        cv.run(data);
        CrossValidation.Score score = cv.getResult();
        System.out.println("Precision: " + score.getPrecision());
        System.out.println("Recall: " + score.getRecall());
        System.out.println("F1: " + score.getF1());
        System.out.println("F2: " + score.getF2());
    }
}
