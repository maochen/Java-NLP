package org.maochen.nlp.classifier.hmm;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 8/5/15.
 */
public class HMMModel implements Serializable {
    public Table<String, String, Double> emission;
    public Table<String, String, Double> transition;
    public Map<String, Double> emissionMin;

    public HMMModel() {
        emission = HashBasedTable.create();
        transition = HashBasedTable.create();
        emissionMin = new HashMap<>();
    }

}
