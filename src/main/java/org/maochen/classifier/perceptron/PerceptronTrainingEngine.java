package org.maochen.classifier.perceptron;

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
    private static final int MAX_ITERATION = Integer.MAX_VALUE;

    public static PerceptronModel train(List<Tuple> data) {
        PerceptronModel model = new PerceptronModel();
        model.weights = new double[data.get(0).featureVector.length];

        int errCount;
        int iter = 0;

        do {
            LOG.debug("Iteration " + (++iter));
            errCount = data.size();
            for (Tuple entry : data) {
                double error = onlineTrain(entry.featureVector, Integer.valueOf(entry.label), model); // for Xi
                if (error == 0) {
                    errCount--;
                }
            }

        } while (errCount != 0 && iter < MAX_ITERATION);

        return model;
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
}
