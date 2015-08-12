package org.maochen.nlp.test;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.IClassifier;
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
        int round;
        String label;
        int truePos = 0;
        int trueNeg = 0;
        int falsePos = 0;
        int falseNeg = 0;

        public double getF1() {
            double precision = getPrecision();
            double recall = getRecall();
            return 2 * precision * recall / (precision + recall);
        }

        public double getPrecision() {
            return truePos / (double) (truePos + falsePos);
        }

        public double getRecall() {
            return truePos / (double) (truePos + falseNeg);
        }

        public double getAccurancy() {
            return (trueNeg + truePos) / (double) (truePos + trueNeg + falseNeg + falsePos);
        }
    }

    private int round;

    private IClassifier classifier;

    private Set<String> labels;
    private Set<Score> scores = new HashSet<>();

    private boolean shuffledata;


    /**
     * Cross validation.
     *
     * @param data whole testing data collection
     */
    public void run(final List<Tuple> data) {
        List<Tuple> data1 = new ArrayList<>(data);
        this.labels = data.parallelStream().map(x -> x.label).collect(Collectors.toSet());
        if (shuffledata) {
            Collections.shuffle(data1);
        }

        int chunkSize = data.size() / round;

        int reminder = data.size() % chunkSize;

        for (int i = data.size() - 1; i > data.size() - 1 - reminder; i--) {
            LOG.info("Dropping the tail id: " + data.get(i).id);
        }

        for (int i = 0; i < round; i++) {
            List<Tuple> testing = data.subList(i, i + chunkSize);
            List<Tuple> training = data.subList(0, i);
            training.addAll(data.subList(i + chunkSize, data.size()));

            eval(training, testing, i);
        }
    }

    // This is for one round.
    private void eval(List<Tuple> training, List<Tuple> testing, int round) {
        classifier.train(training);

        for (Tuple tuple : testing) {
            String actual = classifier.predict(tuple).entrySet().stream()
                    .max((e1, e2) -> e1.getValue().compareTo(e2.getValue()))
                    .map(Map.Entry::getKey).orElse(StringUtils.EMPTY);
            updateScore(tuple, actual, round);
        }
    }

    private void updateScore(Tuple testingTuple, String actual, int round) {

        labels.stream().forEach(label -> { // Init score.
            Score score = new Score();
            score.round = round;
            score.label = label;
            scores.add(score);
        });

        if (testingTuple.label.equals(actual)) { // Correct Predicted
            scores.stream().filter(x -> x.round == round)
                    .filter(x -> x.label.equals(testingTuple.label))
                    .forEach(score -> score.truePos += 1);

            scores.stream().filter(x -> x.round == round)
                    .filter(x -> !x.label.equals(testingTuple.label))
                    .forEach(score -> score.trueNeg += 1);
        } else { // Wrong predicted
            String wrongLabel = actual;
            String correctLabel = testingTuple.label;

            scores.stream().filter(x -> x.round == round)
                    .filter(x -> x.label.equals(wrongLabel))
                    .forEach(score -> score.falsePos += 1);

            scores.stream().filter(x -> x.round == round)
                    .filter(x -> !x.label.equals(correctLabel))
                    .forEach(score -> score.falseNeg += 1);

            scores.stream().filter(x -> x.round == round) //Rest
                    .filter(x -> !x.label.equals(correctLabel) && !x.label.equals(wrongLabel))
                    .forEach(score -> score.trueNeg += 1);
        }
    }

    /**
     * Constructor
     *
     * @param round       iterations for the cross validation.
     * @param classifier  the actual classifier need to test.
     * @param shuffleData whether the data needs to be shuffled at the begining of the whole
     *                    process.
     */
    public CrossValidation(final int round, final IClassifier classifier,
                           final boolean shuffleData) {
        this.round = round;
        this.classifier = classifier;
        this.shuffledata = shuffleData;
    }
}
