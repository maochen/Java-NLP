package org.maochen.nlp.ml.classifier.maxent;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

import opennlp.model.DataIndexer;

import org.maochen.nlp.ml.datastructure.Tuple;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Maochen on 8/7/15.
 */
@Deprecated // WIP
public class RealValueDataIndexer implements DataIndexer {

    // Id, feat name, occurance
    private BiMap<Integer, String> featsIndexer = HashBiMap.create();
    private Map<Integer, Integer> featsOccurance = new HashMap<>();

    private List<Tuple> trainingData;

    public RealValueDataIndexer(List<Tuple> trainingData) {
        this.trainingData = trainingData;
        generateFeats(trainingData);
    }

    private void generateFeats(List<Tuple> trainingData) {
        int featCount = 0;
        for (Tuple t : trainingData) {
            for (String featName : t.featureName) {
                if (featsIndexer.containsValue(featName)) {
                    int id = featsIndexer.inverse().get(featName);
                    int val = featsOccurance.get(id) + 1;
                    featsOccurance.put(id, val);
                } else {
                    featCount++;
                    featsIndexer.put(featCount, featName);
                    featsOccurance.put(featCount, 1);
                }
            }
        }
    }

    @Override
    public int[][] getContexts() {
        return new int[0][];
    }

    @Override
    public int[] getNumTimesEventsSeen() {
        return new int[0];
    }

    @Override
    public int[] getOutcomeList() {
        return new int[0];
    }

    private String[] predLabels = null;

    @Override
    public String[] getPredLabels() {
        if (predLabels == null) {
            predLabels = featsIndexer.entrySet().stream()
                    .sorted((e1, e2) -> e1.getKey().compareTo(e2.getKey()))
                    .map(Map.Entry::getValue)
                    .toArray(String[]::new);
        }
        return predLabels;
    }

    private int[] predCounts = null;

    @Override
    public int[] getPredCounts() {
        if (predCounts == null) {
            predCounts = new int[featsIndexer.keySet().size()];
            String[] predLabels = getPredLabels();
            for (int i = 0; i < predLabels.length; i++) {
                int id = featsIndexer.inverse().get(predLabels[i]);
                int occurance = featsOccurance.get(id);
                predCounts[i] = occurance;
            }

        }
        return predCounts;
    }

    @Override
    public String[] getOutcomeLabels() {
        return null;
    }

    @Override
    public float[][] getValues() {
        return new float[0][];
    }

    @Override
    public int getNumEvents() {
        return 0;
    }
}
