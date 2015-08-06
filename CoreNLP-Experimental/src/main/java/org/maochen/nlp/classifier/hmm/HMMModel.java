package org.maochen.nlp.classifier.hmm;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMMModel {
    public Table<String, String, Double> emission;
    public Table<String, String, Double> transition;
    public Map<String, Double> emissionMin;

    // TODO: Serialization. Load
    public HMMModel(String file) {
        throw new NotImplementedException();
    }

    public HMMModel() {
        emission = HashBasedTable.create();
        transition = HashBasedTable.create();
        emissionMin = new HashMap<>();
    }

}
