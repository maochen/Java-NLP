package org.maochen.nlp.ml.classifier.perceptron;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.ml.IClassifier;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.util.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 6/5/15.
 */
public class PerceptronClassifier implements IClassifier {

    private static final Logger LOG = LoggerFactory.getLogger(PerceptronClassifier.class);

    protected PerceptronModel model = null;

    private Properties properties = null;

    private static int MAX_ITERATION = 200;

    // Key is LabelIndex.
    private Map<Integer, Double> predict(final double[] x) {
        Map<Integer, Double> result = new HashMap<>();
        for (int i = 0; i < model.weights.length; i++) {
            double y = VectorUtils.dotProduct(x, model.weights[i]);
            y += model.bias[i];
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

    /**
     * public use for doing one training sample.
     *
     * @param x          Feature Vector
     * @param labelIndex label's index, from PerceptronModel.LabelIndexer
     */
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
        }

        if (LOG.isDebugEnabled()) {
            LOG.debug("New bias: " + Arrays.toString(model.bias));
            LOG.debug("New weight: " + Arrays.stream(model.weights).map(Arrays::toString).reduce((wi, wii) -> wi + ", " + wii).get());
        }
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        this.model = new PerceptronModel(trainingData);
        setParameter(properties);

        int errCount;
        int iter = 0;
        do {
            LOG.debug("Iteration " + (++iter));
            Collections.shuffle(trainingData);

            for (Tuple entry : trainingData) {
                onlineTrain(entry.vector.getVector(), model.labelIndexer.getIndex(entry.label)); // for Xi
            }

            errCount = (int) trainingData.stream().filter(entry -> predictMax(entry.vector.getVector()).getLeft() != model.labelIndexer.getIndex(entry.label)).count();
        } while (errCount != 0 && iter < MAX_ITERATION);

        LOG.debug("Err size: " + errCount);
        return this;
    }

    /**
     * Do a prediction.
     *
     * @param predict predict tuple.
     * @return Map's key is the actual label, Value is probability, the probability is random depends on the order of training
     * sample.
     */
    @Override
    public Map<String, Double> predict(Tuple predict) {
        Map<Integer, Double> indexResult = predict(predict.vector.getVector());
        return indexResult.entrySet().stream()
                .map(e -> new ImmutablePair<>(model.labelIndexer.getLabel(e.getKey()), VectorUtils.sigmoid.apply(e.getValue()))) // Only do sigmoid here!
                .collect(Collectors.toMap(ImmutablePair::getLeft, ImmutablePair::getRight));
    }

    @Override
    public void setParameter(Properties props) {
        this.properties = props;

        if (props == null) {
            return;
        }

        if (props.containsKey("learning_rate")) {
            this.model.learningRate = Double.parseDouble(props.getProperty("learning_rate"));
        }

        if (props.containsKey("iter")) {
            MAX_ITERATION = Integer.parseInt(props.getProperty("iter"));
        }

        if (props.containsKey("threshold")) {
            this.model.threshold = Double.parseDouble(props.getProperty("threshold"));
        }
    }

    @Override
    public void persistModel(String modelFile) throws IOException {
        model.persist(modelFile);
    }

    @Override
    public void loadModel(InputStream inputStream) {
        model.load(inputStream);
    }

    public PerceptronClassifier() {
        this.model = new PerceptronModel();
    }
}
