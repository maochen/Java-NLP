package org.maochen.nlp.datastructure;

import org.apache.commons.lang3.StringUtils;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 12/3/14.
 */
public class Tuple {
    public int id;
    public String label;

    public String[] featureName = null; //Optional
    public double[] featureVector;

    private Map<String, Object> extra = null;

    // This is for predict
    public Tuple(double[] featureVector) {
        this.featureVector = featureVector;
        label = null;
    }

    // This is for training data
    public Tuple(int id, double[] featureVector, String label) {
        this.id = id;
        this.featureVector = featureVector;
        this.label = label;
    }

    public Tuple(int id, String[] featureName, double[] featureVector, String label) {
        this.id = id;
        this.featureName = featureName;
        this.featureVector = featureVector;
        this.label = label;
    }

    public Map<String, Object> getExtra() {
        return extra;
    }

    public void addExtra(String key, Object val) {
        if (extra == null) {
            extra = new HashMap<>();
        }
        extra.put(key, val);
    }


    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(label).append(StringUtils.SPACE);
        for (double v : featureVector) {
            stringBuilder.append(v).append(StringUtils.SPACE);
        }

        return stringBuilder.toString().trim();
    }
}
