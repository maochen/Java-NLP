package org.maochen.nlp.app.chunker;

import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
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

    private static void addFeat(Set<String> feat, String key, String... val) {
        String entry = Arrays.stream(val).reduce((v1, v2) -> v1 + "_" + v2).get();
        entry = key + "=" + entry;
        feat.add(entry);
    }

    public static Set<String> extractFeatSingle(int i, final String[] tokens, final String[] pos, final String[] resolvedPrevTags) {
        Set<String> currentFeats = new HashSet<>();

        addFeat(currentFeats, "w0", tokens[i]);
        addFeat(currentFeats, "pos0", pos[i]);
        if (i > 0) {
            addFeat(currentFeats, "tag-1", resolvedPrevTags[i - 1]);
            addFeat(currentFeats, "pos-1tag-1pos0", pos[i - 1], resolvedPrevTags[i - 1], pos[i]);

            addFeat(currentFeats, "w-1", tokens[i - 1]);
            addFeat(currentFeats, "pos-1", pos[i - 1]);
        }

        if (i > 1) {
            addFeat(currentFeats, "tag-2", resolvedPrevTags[i - 2]);
            addFeat(currentFeats, "tag-2-1pos0", resolvedPrevTags[i - 2], resolvedPrevTags[i - 1], pos[i]);

            addFeat(currentFeats, "pos-2", pos[i - 2]);
            addFeat(currentFeats, "pos-2-1", pos[i - 2], pos[i - 1]);
            addFeat(currentFeats, "pos-20", pos[i - 2], pos[i]);
        }

        if (i > 2) {
            addFeat(currentFeats, "tag-3", resolvedPrevTags[i - 3]);
            addFeat(currentFeats, "tag-3-2-1pos0", resolvedPrevTags[i - 3], resolvedPrevTags[i - 2], resolvedPrevTags[i - 1], pos[i]);

            addFeat(currentFeats, "pos-3", pos[i - 3]);
            addFeat(currentFeats, "pos-30", pos[i - 3], pos[i]);
            addFeat(currentFeats, "pos-3-2", pos[i - 3], pos[i - 2]);
        }

        if (i < tokens.length - 1) {
            addFeat(currentFeats, "w+1", tokens[i + 1]);
            addFeat(currentFeats, "pos+1", pos[i + 1]);
            addFeat(currentFeats, "pos0+1", pos[i], pos[i + 1]);
        }

        if (i < tokens.length - 2) {
            addFeat(currentFeats, "pos+2", pos[i + 2]);
            addFeat(currentFeats, "pos0+2", pos[i], pos[i + 2]);
            addFeat(currentFeats, "pos0+1+2", pos[i], pos[i + 1], pos[i + 2]);
        }

        if (i > 0 && i < tokens.length - 1) {
            addFeat(currentFeats, "pos-10+1", pos[i - 1], pos[i], pos[i + 1]);
        }

        if (i > 1 && i < tokens.length - 1) {
            addFeat(currentFeats, "pos-2-10+1", pos[i - 2], pos[i - 1], pos[i], pos[i + 1]);
        }


        return currentFeats;
    }

    /**
     * Single SequenceTuple featExtractor
     */
    public static List<Tuple> extractFeat(final SequenceTuple entry) {
        String[] tokens = entry.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[Chunker.WORD_INDEX]).toArray(String[]::new);
        String[] pos = entry.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[Chunker.POS_INDEX]).toArray(String[]::new);

        List<Set<String>> feats = IntStream.range(0, tokens.length)
                .mapToObj(i -> extractFeatSingle(i, tokens, pos, entry.tag.stream().toArray(String[]::new)))
                .collect(Collectors.toList());

        List<Tuple> tuples = new ArrayList<>();

        for (int i = 0; i < feats.size(); i++) {
            Set<String> singleTokenFeat = feats.get(i);
            LabeledVector v = new LabeledVector(singleTokenFeat.stream().toArray(String[]::new));

            Tuple t = new Tuple(v);
            t.label = entry.tag.get(i);

            tuples.add(t);
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
