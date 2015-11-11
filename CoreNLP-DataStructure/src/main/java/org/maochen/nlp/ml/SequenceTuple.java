package org.maochen.nlp.ml;

import org.maochen.nlp.ml.vector.LabeledVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * This tuple is for the Sequence Model using. Don't mess up with regular Tuple.
 *
 * Created by Maochen on 8/5/15.
 */
public class SequenceTuple {
    public int id;

    /*
      Tuple1, Tuple2,  Tuple3,   Tuple4
        I,     have,     a,        car
        PRP,   VBP,      DT,       NN
  tag:   O,    B-VP,    B-NP,     I-NP
     */
    public List<Tuple> entries;
    public List<String> tag;

    public SequenceTuple(Map<Integer, List<String>> entries, List<String> tags) {
        if (entries == null || tags == null || entries.get(0).size() != tags.size()) {
            throw new RuntimeException("words and tags are invalid (Size mismatch).");
        }

        this.entries = new ArrayList<>();

        int max = entries.keySet().stream().max(Integer::compare).get();

        for (int i = 0; i <= max; i++) {
            Tuple entry = null;

            if (entries.containsKey(i)) {
                entry = new Tuple(new LabeledVector(entries.get(i).stream().toArray(String[]::new)));
                entry.label = tags.get(i);
            }

            this.entries.add(entry);
        }

        this.tag = tags;
    }

    public SequenceTuple() {

    }
}
