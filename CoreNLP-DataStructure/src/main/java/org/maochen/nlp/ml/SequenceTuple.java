package org.maochen.nlp.ml;

import org.maochen.nlp.ml.vector.IVector;
import org.maochen.nlp.ml.vector.LabeledVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This tuple is for the Sequence Model using. Don't mess up with regular Tuple.
 *
 * Created by Maochen on 8/5/15.
 */
public class SequenceTuple {

    public int id;
    public List<Tuple> entries;

    /**
     * featMap
     *
     * 0,         [ I,     have,    a,      car]
     *
     * 1,         null
     *
     * 2,         [PRP,     VBP,    DT,     NN]
     *
     * tags       [B-NP,   B-VP,   B-NP,   I-NP]
     *
     * -------------------------------------------
     *
     * entries   Tuple1, Tuple2, Tuple3, Tuple4
     */
    public SequenceTuple(Map<Integer, List<String>> featMap, List<String> tags) {
        if (featMap == null || tags == null || featMap.values().stream().findFirst().get().size() != tags.size()) {
            throw new RuntimeException("words and tags are invalid (Size mismatch).");
        }

        int[] dimensions = featMap.values().stream().mapToInt(List::size).distinct().toArray();
        if (dimensions.length != 1) {
            throw new RuntimeException("feats dimension size mismatch).");
        }

        int featIndexMax = featMap.keySet().stream().max(Integer::compare).get();

        String[][] matrix = new String[featIndexMax + 1][];

        featMap.entrySet().forEach(entry -> matrix[entry.getKey()] = entry.getValue().stream().toArray(String[]::new));

        List<Tuple> tuples = new ArrayList<>(dimensions[0]);

        for (int col = 0; col < dimensions[0]; col++) {
            List<String> featString = new ArrayList<>();
            for (int row = 0; row < matrix.length; row++) {
                featString.add(matrix[row][col]);
            }

            IVector v = new LabeledVector(featString.stream().toArray(String[]::new));
            tuples.add(new Tuple(0, v, tags.get(col)));
        }

        this.entries = tuples;
    }

    public List<String> getLabel() {
        return this.entries.stream().map(x -> x.label).collect(Collectors.toList());
    }

    public SequenceTuple() {

    }
}
