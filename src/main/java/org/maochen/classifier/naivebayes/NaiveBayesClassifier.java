package org.maochen.classifier.naivebayes;

import org.maochen.classifier.IClassifier;
import org.maochen.datastructure.Tuple;
import org.maochen.datastructure.LabelIndexer;
import org.maochen.utils.TupleUtils;
import org.maochen.utils.VectorUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Maochen on 12/3/14.
 */
public class NaiveBayesClassifier implements IClassifier {

    private TrainingEngine trainingEngine;

    private int labelSize = -1;
    private int featureLength = -1;

    private LabelIndexer labelIndexer = new LabelIndexer();

    /**
     * @param paraMap
     */
    @Override
    public void setParameter(Map<String, String> paraMap) {
        if (paraMap.containsKey("label_size")) {
            labelSize = Integer.parseInt(paraMap.get("label_size"));
        }

        if (paraMap.containsKey("feature_len")) {
            featureLength = Integer.parseInt(paraMap.get("feature_len"));
        }
    }

    // Integer is label's Index from labelIndexer
    @Override
    public Map<String, Double> predict(Tuple predict) {
        Map<Integer, Double> probs = new HashMap<>();

        for (String label : labelIndexer.getAllLabels()) {
            int index = labelIndexer.getIndex(label);
            // P(male) -- P(C)
            probs.put(index, 0.5);
        }

        double maxProb = 0;
        int maxProbLabel = -1;
        for (int i = 0; i < predict.featureVector.length; i++) {
            double fi = predict.featureVector[i];

            for (String label : labelIndexer.getAllLabels()) {
                int labelIndex = labelIndexer.getIndex(label);

                double newProb = probs.get(labelIndex) * VectorUtils.gaussianDensityDistribution(trainingEngine.meanVectors[labelIndex][i], trainingEngine.varianceVectors[labelIndex][i], fi);
                probs.put(labelIndex, newProb);
                if (newProb > maxProb) {
                    maxProb = newProb;
                    maxProbLabel = labelIndex;
                }
            }
        }

        predict.label = labelIndexer.getLabel(maxProbLabel);

        return TupleUtils.convertMap(probs, labelIndexer);
    }

    @Override
    public IClassifier train(List<Tuple> trainingData) {
        if (labelSize <= 0 || featureLength <= 0) {
            throw new RuntimeException("Label size and feature length are required.");
        }

        trainingEngine = new TrainingEngine(labelSize, featureLength);
        trainingEngine.init(trainingData, labelIndexer);
        trainingEngine.calculateMean();
        trainingEngine.calculateVariance();
        return this;
    }

    public static void main(String[] args) {
        IClassifier nbc = new NaiveBayesClassifier();
        Map<String, String> paraMap = new HashMap<>();
        paraMap.put("label_size", "2");
        paraMap.put("feature_len", "3");
        nbc.setParameter(paraMap);

        List<Tuple> trainingData = new ArrayList<>();
        trainingData.add(new Tuple(1, new double[]{6, 180, 12}, "male"));
        trainingData.add(new Tuple(2, new double[]{5.92, 190, 11}, "male"));
        trainingData.add(new Tuple(3, new double[]{5.58, 170, 12}, "male"));
        trainingData.add(new Tuple(4, new double[]{5.92, 165, 10}, "male"));
        trainingData.add(new Tuple(5, new double[]{5, 100, 6}, "female"));
        trainingData.add(new Tuple(6, new double[]{5.5, 150, 8}, "female"));
        trainingData.add(new Tuple(7, new double[]{5.42, 130, 7}, "female"));
        trainingData.add(new Tuple(8, new double[]{5.75, 150, 9}, "female"));

        Tuple predict = new Tuple(new double[]{6, 130, 8});

        nbc.train(trainingData);
        Map<String, Double> probs = nbc.predict(predict);

        // double expectedMaleProb = probs.get("male");
        // double expectedFemaleProb = probs.get("female");
        // assert expectedMaleProb == 6.197071843878083E-9;
        // assert expectedFemaleProb == 5.377909183630024E-4;


        System.out.println("Result: " + predict);
        System.out.println(probs);
    }


}
