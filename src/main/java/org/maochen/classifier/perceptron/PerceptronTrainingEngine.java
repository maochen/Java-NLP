package org.maochen.classifier.perceptron;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;
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
    //    private static final double SQUARELOSS_THRESHOLD = 0.3;

    private static Predicate<Object[]> shouldTerminate = x -> {
        if ((int) x[0] == 0) {
            return true;
        }

        if ((int) x[1] > MAX_ITERATION) {
            return true;
        }

        //        if ((double) x[2] < SQUARELOSS_THRESHOLD) {
        //            return true;
        //        }

        return false;
    };

    public static PerceptronModel train(List<Tuple> trainingData, PerceptronClassifier perceptronClassifier) {
        PerceptronModel model = new PerceptronModel();
        model.trainBias = perceptronClassifier.trainBias;
        model.weights = new double[trainingData.stream().findFirst().orElse(null).featureVector.length];
        perceptronClassifier.model = model;

        int errCount;
        int iter = 0;
        //        double squareloss = 0D;
        do {
            LOG.debug("Iteration " + (++iter));
            errCount = trainingData.size();
            for (Tuple entry : trainingData) {
                Pair<Integer, PerceptronModel> result = onlineTrain(entry.featureVector, Integer.valueOf(entry.label), model); // for Xi
                model = result.getRight();

                //                squareloss = 0.5 * Math.pow(perceptronClassifier.predict(entry).get(entry.label) - Integer.valueOf(entry.label), 2);
                //                LOG.debug("Square loss: " + squareloss);
                if (result.getLeft() == 0) {
                    errCount--;
                }
            }
        } while (!shouldTerminate.test(new Object[]{errCount, iter}));

        return model;
    }


    // public use for doing one training sample.
    // instead of directly change a model, we will do a copy of a model and change the copy.
    // return error and copy of perceptron model
    public static Pair<Integer, PerceptronModel> onlineTrain(final double[] x, final int labelIndex, final PerceptronModel originalModel) {
        PerceptronModel model = new PerceptronModel(originalModel);

        double sum = VectorUtils.dotProduct(x, model.weights);
        sum += model.bias * labelIndex;
        int network = sum > model.threshold ? 1 : 0;
        int error = labelIndex - network; // might be negative. gold -> 1. actual -> 0

        if (error != 0) {
            double correction_d = model.learningRate * error; // err(-1,1) controls +/-
            model.weights = IntStream.range(0, model.weights.length)
                    .mapToDouble(i -> model.weights[i] + x[i] * correction_d)
                    .toArray();
            if (model.trainBias) {
                model.bias = model.bias + labelIndex * correction_d;
            }
        }

        LOG.debug("New weights: " + Arrays.toString(model.weights) + " | New bias: " + model.bias);
        return new ImmutablePair<>(error, model);
    }
}
