package org.maochen.nlp.app.chunker;

import org.apache.commons.lang3.NotImplementedException;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Chunking with Maximum Entropy Models.
 *
 * http://www.cnts.ua.ac.be/conll2000/pdf/13941koe.pdf
 *
 * <p>Created by Maochen on 11/10/15.
 */
public class ChunkerFeatureExtractor {

    private static List<Set<String>> extractFeatInternal(final String[] tokens, final String[] pos) {
        throw new NotImplementedException("");
    }

    /**
     * Single SequenceTuple featExtractor
     */
    public static List<Tuple> extractFeat(final SequenceTuple entry) {
        String[] tokens = ((LabeledVector) entry.entries.get(Chunker.WORD_INDEX).vector).featsName;
        String[] pos = ((LabeledVector) entry.entries.get(Chunker.POS_INDEX).vector).featsName;

        List<Set<String>> feats = extractFeatInternal(tokens, pos);

        List<Tuple> tuples = new ArrayList<>();

        for (int i = 0; i < feats.size(); i++) {
            Set<String> singleTokenFeat = feats.get(i);
            double[] paddingVal = IntStream.range(0, singleTokenFeat.size()).asDoubleStream().map(x -> 1D).toArray();
            LabeledVector v = new LabeledVector(paddingVal);
            v.featsName = singleTokenFeat.stream().toArray(String[]::new);

            Tuple t = new Tuple(v);
            t.label = pos[i];
        }

        return tuples;
    }


    /**
     * pos is in the extra feats, key is word id.
     *
     * return list only for compatible with MaxEnt interface. Not necessary to be list in logic.
     */
    public static List<Tuple> extract(final Set<SequenceTuple> trainingData) {
        if (trainingData == null) {
            return null;
        }

        return trainingData.parallelStream().map(ChunkerFeatureExtractor::extractFeat).flatMap(Collection::stream).collect(Collectors.toList());
    }
}
