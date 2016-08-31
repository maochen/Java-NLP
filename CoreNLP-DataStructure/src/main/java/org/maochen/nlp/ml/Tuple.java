package org.maochen.nlp.ml;

import org.apache.commons.lang3.StringUtils;
import org.maochen.nlp.ml.vector.DenseVector;
import org.maochen.nlp.ml.vector.IVector;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Maochen on 12/3/14.
 */
public class Tuple {
    public int id;
    public String label;
    public IVector vector;

    private Map<String, Object> extra = null;

    // This is for predict
    public Tuple(IVector vector) {
        this.vector = vector;
        this.label = null;
    }

    public Tuple(int id, IVector vector, String label) {
        this.id = id;
        this.vector = vector;
        this.label = label;
    }

    /**
     * Default using dense vector.
     */
    public Tuple(double[] featureVector) {
        this.vector = new DenseVector(featureVector);
        label = null;
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
        stringBuilder.append(vector);

        return stringBuilder.toString().trim();
    }
}
