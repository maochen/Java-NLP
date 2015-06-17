package org.maochen.classifier.perceptron;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private static final int MAX_ITERATION = 2000;

    public static PerceptronModel train(List<Tuple> trainingData) {
        PerceptronModel model = new PerceptronModel();
        model.weights = new double[trainingData.stream().findFirst().orElse(null).featureVector.length];

        int errCount;
        int iter = 0;
        do {
            LOG.debug("Iteration " + (++iter));
            errCount = trainingData.size();
            for (Tuple entry : trainingData) {
                Pair<Integer, PerceptronModel> result = onlineTrain(entry.featureVector, Integer.valueOf(entry.label), model); // for Xi
                model = result.getRight();
                if (result.getLeft() == 0) {
                    errCount--;
                }
            }
        } while (errCount != 0 && iter < MAX_ITERATION);

        return model;
    }


    // public use for doing one training sample.
    // instead of directly change a model, we will do a copy of a model and change the copy.
    // return error and copy of perceptron model
    public static Pair<Integer, PerceptronModel> onlineTrain(final double[] x, final int labelIndex, final PerceptronModel originalModel) {
        PerceptronModel model = new PerceptronModel();
        model.learningRate = originalModel.learningRate;
        model.threshold = originalModel.threshold;
        model.weights = originalModel.weights;

        double sum = VectorUtils.dotProduct(x, model.weights);
        int network = sum > model.threshold ? 1 : 0;
        int error = labelIndex - network; // might be negative.

        if (error != 0) {
            double correction_d = model.learningRate * error;
            model.weights = IntStream.range(0, model.weights.length).mapToDouble(i -> model.weights[i] + x[i] * correction_d).toArray();
        }

        LOG.debug("New weights: " + Arrays.toString(model.weights));
        return new ImmutablePair<>(error, model);
    }
}
