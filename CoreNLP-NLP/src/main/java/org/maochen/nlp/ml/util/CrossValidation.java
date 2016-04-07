package org.maochen.nlp.ml.util;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/11/15.
 */
public class CrossValidation {
    private static final Logger LOG = LoggerFactory.getLogger(CrossValidation.class);

    static class Score {
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

        public double getPrecision() {
            return tp / (double) (tp + fp);
        }

        public double getRecall() {
            return tp / (double) (tp + fn);
        }

        public double getAccuracy() {
            return (tn + tp) / (double) (tp + tn + fn + fp);
        }
    }

    private int nfold;

    private IClassifier classifier;

    private Set<String> labels;
    private Set<Score> scores = new HashSet<>();

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
            List<Tuple> testing = data.subList(i, i + chunkSize);
            List<Tuple> training = data.subList(0, i);
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
        labels.stream().forEach(label -> { // Init score.
            Score score = new Score();
            score.nfold = nfold;
            score.label = label;
            scores.add(score);
        });

        if (testingTuple.label.equals(actual)) { // Correct Predicted
            scores.stream().filter(x -> x.nfold == nfold)
                    .filter(x -> x.label.equals(testingTuple.label))
                    .forEach(score -> score.tp += 1);

            scores.stream().filter(x -> x.nfold == nfold)
                    .filter(x -> !x.label.equals(testingTuple.label))
                    .forEach(score -> score.tn += 1);
        } else { // Wrong predicted
            String wrongLabel = actual;
            String correctLabel = testingTuple.label;

            scores.stream().filter(x -> x.nfold == nfold)
                    .filter(x -> x.label.equals(wrongLabel))
                    .forEach(score -> score.fp += 1);

            scores.stream().filter(x -> x.nfold == nfold)
                    .filter(x -> !x.label.equals(correctLabel))
                    .forEach(score -> score.fn += 1);

            scores.stream().filter(x -> x.nfold == nfold) //Rest
                    .filter(x -> !x.label.equals(correctLabel) && !x.label.equals(wrongLabel))
                    .forEach(score -> score.tn += 1);
        }
    }

    /**
     * Constructor
     *
     * @param nfold       nfold for the cross validation. (Recommand: 10-fold)
     * @param classifier  the actual classifier need to test.
     * @param shuffleData whether the data needs to be shuffled at the begining of the whole
     *                    process.
     */
    public CrossValidation(final int nfold, final IClassifier classifier,
                           final boolean shuffleData) {
        this.nfold = nfold;
        this.classifier = classifier;
        this.shuffleData = shuffleData;
    }
}
