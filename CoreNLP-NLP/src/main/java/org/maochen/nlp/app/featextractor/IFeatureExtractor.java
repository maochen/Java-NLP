package org.maochen.nlp.app.featextractor;

import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;

import java.util.Arrays;
import java.util.List;

/**
 * Created by Maochen on 2/22/16.
 */
public interface IFeatureExtractor {
    List<Tuple> extractFeat(final SequenceTuple entry);

    static void addFeat(List<String> feat, String key, String... val) {
        String entry = Arrays.stream(val).reduce((v1, v2) -> v1 + "_" + v2).orElse(null);

        if (entry == null) {
            entry = key;
        } else {
            entry = key + "=" + entry;
        }
        feat.add(entry);
    }
}
