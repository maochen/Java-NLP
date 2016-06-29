package org.maochen.nlp.ml.classifier;

import com.google.common.collect.ImmutableSet;
import org.maochen.nlp.ml.util.dataio.CSVDataReader;
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
                String[] featFields = featureVector.featsName[j].split("=");
                String featName = featFields[0];
                double featVal = featFields.length > 1 ? featureVector.getVector()[j] : Double.parseDouble(featFields[1]);
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

    public static void main(String[] args) {
        String filename = "";
        CSVDataReader csvDataReader = new CSVDataReader(filename, 6, "\t", true, ImmutableSet.of(1, 2, 3, 4,9), 0);
    }
}
