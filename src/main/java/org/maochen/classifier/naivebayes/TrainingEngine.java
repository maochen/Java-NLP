package org.maochen.classifier.naivebayes;

import org.maochen.datastructure.Element;
import org.maochen.datastructure.LabelIndexer;
import org.maochen.utils.VectorUtils;

import java.util.*;

/**
 * Created by Maochen on 12/3/14.
 */
final class TrainingEngine {

    LabelIndexer labelIndexer;

    List<Element> trainingData;

    // row=labelSize,col=featureLength
    double[][] meanVectors;
    double[][] varianceVectors;
    int[] count;
    int vectorLength;
    int labelSize;

    // Step 1
    public void calculateMean() {
        for (Element t : trainingData) {
            int index = labelIndexer.getIndex(t.label);
            count[index]++;
            double[] newMeanVector = VectorUtils.addition(meanVectors[index], t.featureVector);
            meanVectors[index] = newMeanVector;
        }

        for (int i = 0; i < meanVectors.length; i++) {
            double[] meanVector = meanVectors[i];
            meanVector = VectorUtils.scale(meanVector, 1.0 / count[i]);
            meanVectors[i] = meanVector;
        }
    }

    // Step 2
    public void calculateVariance() {
        for (Element t : trainingData) {
            int index = labelIndexer.getIndex(t.label);
            double[] diff = VectorUtils.minus(t.featureVector, meanVectors[index]);
            diff = VectorUtils.multiply(diff, diff);

            double[] varianceVector = VectorUtils.addition(varianceVectors[index], diff);
            varianceVectors[index] = varianceVector;
        }

        for (int i = 0; i < varianceVectors.length; i++) {
            double[] varianceVector = varianceVectors[i];
            // Denominator is Sample Var instead of Population Var
            varianceVector = VectorUtils.scale(varianceVector, 1.0 / (count[i] - 1));
            varianceVectors[i] = varianceVector;
        }
    }

    public void init(List<Element> trainingData, LabelIndexer labelIndexer) {
        this.labelIndexer = labelIndexer;
        this.trainingData = trainingData;

        for (Element element : this.trainingData) {
            if (!labelIndexer.hasLabel(element.label)) {
                labelIndexer.put(element.label);
            }
        }
    }

    public TrainingEngine(int labelSize, int vectorLength) {
        this.labelSize = labelSize;
        this.vectorLength = vectorLength;
        count = new int[labelSize];

        meanVectors = new double[labelSize][vectorLength];
        varianceVectors = new double[labelSize][vectorLength];

        for (int i = 0; i < labelSize; i++) {
            meanVectors[i] = VectorUtils.allXVector(0, vectorLength);
            varianceVectors[i] = VectorUtils.allXVector(0, vectorLength);
        }
    }

}
