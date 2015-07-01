package org.maochen.classifier.perceptron;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Predicate;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronClassifier implements IClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(PerceptronClassifier.class);

    protected PerceptronModel model = null;

    private static final int MAX_ITERATION = 2000;

    private static Predicate<Object[]> shouldTerminate = x -> {
        if ((int) x[0] == 0) {
            return true;
        }

        if ((int) x[1] > MAX_ITERATION) {
            return true;
        }

        return false;
    };

    // Key is LabelIndex.
    private Map<Integer, Double> predict(final double[] x) {
        Map<Integer, Double> result = new HashMap<>();
        for (int i = 0; i < model.weights.length; i++) {
            double y = VectorUtils.dotProduct(x, model.weights[i]);
            y += model.bias[i];
            y = VectorUtils.stochasticBinary.apply(y);// Logistic Function

            result.put(i, y);
        }

        return result;
    }

    private Pair<Integer, Double> predictMax(final double[] x) {
        Map.Entry<Integer, Double> result = predict(x).entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue())).orElse(null);
        return result == null ? null : new ImmutablePair<>(result.getKey(), result.getValue());
    }

    private double[] reweight(final double[] x, final double[] weight, final double correctionD) {
        return IntStream.range(0, x.length)
                .mapToDouble(i -> weight[i] + model.learningRate * correctionD * x[i]) // Main Part, +1 strategy
                .toArray();
    }

    // public use for doing one training sample.
    // instead of directly change a model, we will do a copy of a model and change the copy.
    // Target is labelIndex
    // return error and copy of perceptron model
    public void onlineTrain(final double[] x, final int labelIndex) {
        Map<Integer, Double> result = predict(x);
        Map.Entry<Integer, Double> maxResult = result.entrySet().stream().max((e1, e2) -> e1.getValue().compareTo(e2.getValue())).orElse(null);

        if (maxResult.getKey() != labelIndex) {
            double e_correction_d = 1;
            model.weights[labelIndex] = reweight(x, model.weights[labelIndex], e_correction_d);
            model.bias[labelIndex] = e_correction_d;

            double w_correction_d = -1;
            model.weights[maxResult.getKey()] = reweight(x, model.weights[maxResult.getKey()], w_correction_d);
            model.bias[maxResult.getKey()] = w_correction_d;

            if (LOG.isDebugEnabled()) {
                LOG.debug("New bias: " + Arrays.toString(model.bias));
                LOG.debug("New weight: ");
                LOG.debug(Arrays.stream(model.weights).map(Arrays::toString).reduce((wi, wii) -> wi + System.lineSeparator() + wii).get());
            }
        }
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        model = new PerceptronModel(trainingData);

        int errCount;
        int iter = 0;
        do {
            LOG.debug("Iteration " + (++iter));
            errCount = trainingData.size();
            for (Tuple entry : trainingData) {
                onlineTrain(entry.featureVector, model.labelIndexer.getIndex(entry.label)); // for Xi

                if (predictMax(entry.featureVector).getLeft() == model.labelIndexer.getIndex(entry.label)) {
                    errCount--;
                }
            }
        } while (!shouldTerminate.test(new Object[]{errCount, iter}));

        return this;
    }

    @Override
    public Map<String, Double> predict(Tuple predict) {
        Map<Integer, Double> indexResult = predict(predict.featureVector);
        Map<String, Double> result = new HashMap<>();
        for (Integer index : indexResult.keySet()) {
            result.put(model.labelIndexer.getLabel(index), indexResult.get(index));
        }
        return result;
    }

    @Override
    public void setParameter(Map<String, String> paraMap) {

    }

    public PerceptronClassifier() {
        this.model = new PerceptronModel();
    }

    public static void main(String[] args) throws FileNotFoundException {
        String modelPath = "/Users/Maochen/Desktop/perceptron_model.dat";
        PerceptronClassifier perceptronClassifier = new PerceptronClassifier();

        List<Tuple> data = new ArrayList<>();
        data.add(new Tuple(1, new double[]{1, 0, 0}, String.valueOf(1)));
        data.add(new Tuple(2, new double[]{1, 0, 1}, String.valueOf(1)));
        data.add(new Tuple(3, new double[]{1, 1, 0}, String.valueOf(1)));
        data.add(new Tuple(4, new double[]{1, 1, 1}, String.valueOf(0)));
        perceptronClassifier.train(data);

        perceptronClassifier.model.persist(modelPath);
        perceptronClassifier = new PerceptronClassifier();
        perceptronClassifier.model.load(new FileInputStream(modelPath));

        Tuple test = new Tuple(5, new double[]{1, 1, 1}, null);
        System.out.println(perceptronClassifier.predict(test));
    }
}
