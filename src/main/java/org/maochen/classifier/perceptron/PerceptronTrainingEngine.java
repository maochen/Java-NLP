package org.maochen.classifier.perceptron;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * This only applies to binary.
 * http://en.wikipedia.org/wiki/Perceptron
 * <p>
 * Created by Maochen on 6/5/15.
 */
public class PerceptronTrainingEngine {

    private static final Logger LOG = LoggerFactory.getLogger(PerceptronTrainingEngine.class);

    private PerceptronModel model;

    private int maxIteration = Integer.MAX_VALUE;

    private double offlineLearningErrThreshold = 0.001;

    public void train(List<Pair<double[], Integer>> data) {
        model = new PerceptronModel();
        model.weights = new double[data.get(0).getLeft().length];

        int errCount;
        int iter = 0;

        do {
            LOG.debug("Iteration " + (++iter));
            errCount = data.size();
            for (Pair<double[], Integer> entry : data) {
                double error = onlineTrain(entry.getLeft(), entry.getRight(), model); // for Xi
                if (error == 0) {
                    errCount--;
                }
            }

        } while (errCount != 0 && iter < maxIteration);
    }


    // public use for doing one training sample.
    // return error
    public static double onlineTrain(double[] x, int labelIndex, PerceptronModel model) {
        double sum = VectorUtils.dotProduct(x, model.weights);
        int network = sum > model.threshold ? 1 : 0;
        int error = labelIndex - network; // might be negative.

        if (error != 0) {
            double correction_d = model.learningRate * error;
            model.weights = IntStream.range(0, model.weights.length).mapToDouble(i -> model.weights[i] + x[i] * correction_d).toArray();
        }

        LOG.debug("New weights: " + Arrays.toString(model.weights));
        return error;
    }

    public static void main(String[] args) {
        PerceptronTrainingEngine te = new PerceptronTrainingEngine();

        List<Pair<double[], Integer>> data = new ArrayList<>();
        data.add(new ImmutablePair<>(new double[]{1, 0, 0}, 1));
        data.add(new ImmutablePair<>(new double[]{1, 0, 1}, 1));
        data.add(new ImmutablePair<>(new double[]{1, 1, 0}, 1));
        data.add(new ImmutablePair<>(new double[]{1, 1, 1}, 0));
        te.train(data);

    }
}
