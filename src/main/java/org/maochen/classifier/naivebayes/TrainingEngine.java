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

    LabelIndexer labelIndexer;
    List<Tuple> trainingData;

    // row=labelSize,col=featureLength
    double[][] meanVectors;
    double[][] varianceVectors;
    int[] count;
    int vectorLength;
    int labelSize;

    // Step 1
    public void calculateMean() {
        for (Tuple t : trainingData) {
            int index = labelIndexer.getIndex(t.label);
            count[index]++;
            meanVectors[index] = VectorUtils.zip(meanVectors[index], t.featureVector, (x, y) -> x + y);
        }

        for (int i = 0; i < meanVectors.length; i++) {
            // Get each label and normalize
            double[] meanVector = meanVectors[i]; // feat(label)
            meanVector = VectorUtils.scale(meanVector, 1.0 / count[i]);
            meanVectors[i] = meanVector;
        }
    }

    // Step 2
    public void calculateVariance() {
        for (Tuple t : trainingData) {
            int index = labelIndexer.getIndex(t.label);
            double[] diff = VectorUtils.zip(t.featureVector, meanVectors[index], (x, y) -> x - y);
            diff = Arrays.stream(diff).map(x -> x * x).toArray();

            double[] varianceVector = VectorUtils.zip(varianceVectors[index], diff, (x, y) -> x + y);
            varianceVectors[index] = varianceVector;
        }

        for (int i = 0; i < varianceVectors.length; i++) {
            double[] varianceVector = varianceVectors[i];
            // Denominator is Sample Var instead of Population Var
            varianceVector = VectorUtils.scale(varianceVector, 1.0 / (count[i] - 1));
            varianceVectors[i] = varianceVector;
        }
    }

    public void init(List<Tuple> trainingData, LabelIndexer labelIndexer) {
        this.labelIndexer = labelIndexer;
        this.trainingData = trainingData;

        this.trainingData.stream()
                .filter(tuple -> !labelIndexer.hasLabel(tuple.label))
                .forEach(tuple -> labelIndexer.putByLabel(tuple.label));
    }

    public TrainingEngine(int labelSize, int vectorLength) {
        this.labelSize = labelSize;
        this.vectorLength = vectorLength;
        count = new int[labelSize];

        meanVectors = new double[labelSize][vectorLength];
        varianceVectors = new double[labelSize][vectorLength];

        for (int i = 0; i < labelSize; i++) {
            meanVectors[i] = new double[vectorLength];
            varianceVectors[i] = new double[vectorLength];
        }
    }
}
