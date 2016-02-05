package org.maochen.nlp.ml.classifier.naivebayes;

import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.classifier.LabelIndexer;
import org.maochen.nlp.util.VectorUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Created by Maochen on 12/3/14.
 */
final class NBTrainingEngine {
    private static final Logger LOG = LoggerFactory.getLogger(NBTrainingEngine.class);

    private List<Tuple> trainingData;

    private NaiveBayesModel model;

    private int[] count; // sum the training data by label

    // Step 1
    private void calculateMean() {
        for (Tuple t : trainingData) {
            int index = model.labelIndexer.getIndex(t.label);
            count[index]++;
            model.meanVectors[index] = VectorUtils.addition(model.meanVectors[index], t.vector.getVector());
//                    VectorUtils.zip(model.meanVectors[index], t.vector.getVector(), (x, y) -> x + y); // Performance
        }

        for (int i = 0; i < model.meanVectors.length; i++) {
            // Get each label and normalize
            double[] meanVector = model.meanVectors[i]; // feat(label)
            VectorUtils.scale(meanVector, 1.0 / count[i]);
            model.meanVectors[i] = meanVector;
        }

        for (int i = 0; i < model.meanVectors.length; i++) {
            for (int j = 0; j < model.meanVectors[i].length; j++) {
                if (model.meanVectors[i][j] == 0) {
                    LOG.warn("mean is 0 for label " + model.labelIndexer.labelIndexer.inverse().get(i) + " at dimension " + j);
                    model.meanVectors[i][j] = Double.MIN_VALUE;
                }
            }
        }
    }

    // Step 2
    private void calculateVariance() {
        for (Tuple t : trainingData) {
            int index = model.labelIndexer.getIndex(t.label);

            double[] meanVector = Arrays.copyOf(model.meanVectors[index], model.meanVectors[index].length);
            VectorUtils.scale(meanVector, -1);
            double[] diff = VectorUtils.addition(t.vector.getVector(), meanVector);
//                    VectorUtils.zip(t.vector.getVector(), model.meanVectors[index], (x, y) -> x - y);
            diff = Arrays.stream(diff).map(x -> x * x).toArray();

            double[] varianceVector = VectorUtils.addition(model.varianceVectors[index], diff);
            //  VectorUtils.zip(model.varianceVectors[index], diff, (x, y) -> x + y);
            model.varianceVectors[index] = varianceVector;
        }

        for (int i = 0; i < model.varianceVectors.length; i++) {
            double[] varianceVector = model.varianceVectors[i];
            // Denominator is Sample Var instead of Population Var
            VectorUtils.scale(varianceVector, 1.0 / (count[i] - 1));
            model.varianceVectors[i] = varianceVector;
        }

        for (int i = 0; i < model.varianceVectors.length; i++) {
            for (int j = 0; j < model.varianceVectors[i].length; j++) {
                if (model.varianceVectors[i][j] == 0) {
                    LOG.warn("variance is 0 for label " + model.labelIndexer.labelIndexer.inverse().get(i) + " at dimension " + j);
                    model.varianceVectors[i][j] = Double.MIN_VALUE;
                }
            }
        }
    }

    // Step 3
    // Assume labels have equal probability. Not depends on the training data size.
    public void calculateLabelPrior() {
        double prior = 1D / model.labelIndexer.getLabelSize();
        model.labelIndexer.getIndexSet().forEach(labelIndex -> model.labelPrior.put(labelIndex, prior));
    }

    public NaiveBayesModel train() {
        calculateMean();
        calculateVariance();
        calculateLabelPrior();
        return model;
    }

    public NBTrainingEngine(List<Tuple> trainingData) {
        this.trainingData = trainingData;
        this.model = new NaiveBayesModel();
        this.model.labelIndexer = new LabelIndexer(trainingData);

        int vectorLength = trainingData.stream().findFirst().map(x -> x.vector.getVector().length).orElse(0);
        count = new int[model.labelIndexer.getLabelSize()];

        model.meanVectors = new double[model.labelIndexer.getLabelSize()][vectorLength];
        model.varianceVectors = new double[model.labelIndexer.getLabelSize()][vectorLength];
        model.labelPrior = new HashMap<>();

        for (int i = 0; i < model.labelIndexer.getLabelSize(); i++) {
            model.meanVectors[i] = new double[vectorLength];
            model.varianceVectors[i] = new double[vectorLength];
        }
    }
}
