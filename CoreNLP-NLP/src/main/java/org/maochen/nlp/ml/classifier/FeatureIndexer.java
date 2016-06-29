package org.maochen.nlp.ml.classifier;

import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.util.*;

/**
 * Created by mguan on 6/28/16.
 */
public class FeatureIndexer {

    private Map<String, Integer> nameIndexMap = new HashMap<>();

    private String[] getNames(List<FeatNamedVector> trainingSamples) {
        Set<String> nameSet = new HashSet<>();
        trainingSamples.forEach(t -> {
            for (String s : t.featsName) {
                String featName = s.split("=")[0];
                nameSet.add(featName);
            }
        });

        return nameSet.stream().toArray(String[]::new);
    }

    public double[][] process(List<FeatNamedVector> trainingSamples) {
        if (trainingSamples == null) {
            throw new RuntimeException("Training samples is null.");
        }

        String[] namesArray = getNames(trainingSamples);
        for (int i = 0; i < namesArray.length; i++) {
            nameIndexMap.put(namesArray[i], i);
        }

        double[][] vectors = new double[trainingSamples.size()][];
        for (int i = 0; i < trainingSamples.size(); i++) {
            FeatNamedVector featureVector = trainingSamples.get(i);
            double[] vector = new double[nameIndexMap.size()];
            for (int j = 0; j < featureVector.featsName.length; j++) {
                String[] featNameFields = featureVector.featsName[j].split("=");
                String featName = featNameFields[0];
                double featVal = featNameFields.length < 2 ? featureVector.getVector()[j] : Double.parseDouble(featNameFields[1]);
                int featIndex = nameIndexMap.get(featName);
                vector[featIndex] = featVal;
            }
            vectors[i] = vector;
        }

        return vectors;
    }

    /**
     * This method should only be used for print or debug purpose.
     *
     * @return Array of feat names. Index corresponding to feat vector's Index.
     */
    public String[] getFeatNames() {
        String[] namesArray = new String[nameIndexMap.size()];

        for (Map.Entry<String, Integer> entry : nameIndexMap.entrySet()) {
            namesArray[entry.getValue()] = entry.getKey();
        }

        return namesArray;
    }
}
