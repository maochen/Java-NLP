package org.maochen.classifier.naivebayes;

import org.maochen.datastructure.LabelIndexer;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;

import java.util.Arrays;
import java.util.List;

/**
 * Created by Maochen on 12/3/14.
 */
final class TrainingEngine {

    private List<Tuple> trainingData;

    private NaiveBayesModel model;

    private int[] count; // sum the training data by label

    // Step 1
    public void calculateMean() {
        for (Tuple t : trainingData) {
            int index = model.labelIndexer.getIndex(t.label);
            count[index]++;
            model.meanVectors[index] = VectorUtils.zip(model.meanVectors[index], t.featureVector, (x, y) -> x + y);
        }

        for (int i = 0; i < model.meanVectors.length; i++) {
            // Get each label and normalize
            double[] meanVector = model.meanVectors[i]; // feat(label)
            meanVector = VectorUtils.scale(meanVector, 1.0 / count[i]);
            model.meanVectors[i] = meanVector;
        }
    }

    // Step 2
    public void calculateVariance() {
        for (Tuple t : trainingData) {
            int index = model.labelIndexer.getIndex(t.label);
            double[] diff = VectorUtils.zip(t.featureVector, model.meanVectors[index], (x, y) -> x - y);
            diff = Arrays.stream(diff).map(x -> x * x).toArray();

            double[] varianceVector = VectorUtils.zip(model.varianceVectors[index], diff, (x, y) -> x + y);
            model.varianceVectors[index] = varianceVector;
        }

        for (int i = 0; i < model.varianceVectors.length; i++) {
            double[] varianceVector = model.varianceVectors[i];
            // Denominator is Sample Var instead of Population Var
            varianceVector = VectorUtils.scale(varianceVector, 1.0 / (count[i] - 1));
            model.varianceVectors[i] = varianceVector;
        }
    }

    public NaiveBayesModel train() {
        calculateMean();
        calculateVariance();
        return model;
    }

    public TrainingEngine(List<Tuple> trainingData) {
        this.trainingData = trainingData;
        this.model = new NaiveBayesModel();
        this.model.labelIndexer = new LabelIndexer(trainingData);

        int vectorLength = trainingData.stream().findFirst().map(x -> x.featureVector.length).orElse(0);
        count = new int[model.labelIndexer.getLabelSize()];

        model.meanVectors = new double[model.labelIndexer.getLabelSize()][vectorLength];
        model.varianceVectors = new double[model.labelIndexer.getLabelSize()][vectorLength];

        for (int i = 0; i < model.labelIndexer.getLabelSize(); i++) {
            model.meanVectors[i] = new double[vectorLength];
            model.varianceVectors[i] = new double[vectorLength];
        }
    }
}
