package org.maochen.nlp.app.chunker;

import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

    public static final int WORD_INDEX = 0;
    public static final int POS_INDEX = 1;

    private static void addFeat(List<String> feat, String key, String... val) {
        String entry = Arrays.stream(val).reduce((v1, v2) -> v1 + "_" + v2).get();
        entry = key + "=" + entry;
        feat.add(entry);
    }

    public static List<String> extractFeatSingle(int i, final String[] tokens, final String[] pos, final String[] resolvedPrevTags) {
        List<String> currentFeats = new ArrayList<>();

        addFeat(currentFeats, "w0", tokens[i]);
        addFeat(currentFeats, "pos0", pos[i]);

        if (i > 0) {
            if (resolvedPrevTags != null) {
                addFeat(currentFeats, "tag-1", resolvedPrevTags[i - 1]);
                addFeat(currentFeats, "pos-1tag-1pos0", pos[i - 1], resolvedPrevTags[i - 1], pos[i]);
            }

            addFeat(currentFeats, "w-1", tokens[i - 1]);
            addFeat(currentFeats, "w-10", tokens[i - 1], tokens[i]);
            addFeat(currentFeats, "pos-1", pos[i - 1]);
            addFeat(currentFeats, "pos-10", pos[i - 1], pos[i]);

        }

        if (i > 1) {
            if (resolvedPrevTags != null) {
                addFeat(currentFeats, "tag-2", resolvedPrevTags[i - 2]);
                addFeat(currentFeats, "tag-2-1pos0", resolvedPrevTags[i - 2], resolvedPrevTags[i - 1], pos[i]);
            }

            addFeat(currentFeats, "w-2", tokens[i - 2]);
            addFeat(currentFeats, "pos-2", pos[i - 2]);
            addFeat(currentFeats, "pos-2-1", pos[i - 2], pos[i - 1]);
            addFeat(currentFeats, "pos-20", pos[i - 2], pos[i]);
            addFeat(currentFeats, "pos-2-10", pos[i - 2], pos[i - 1], pos[i]);

        }

        if (i > 2) {
            if (resolvedPrevTags != null) {
                addFeat(currentFeats, "tag-3", resolvedPrevTags[i - 3]);
                addFeat(currentFeats, "tag-3-2-1pos0", resolvedPrevTags[i - 3], resolvedPrevTags[i - 2], resolvedPrevTags[i - 1], pos[i]);
            }

            addFeat(currentFeats, "pos-3", pos[i - 3]);
            addFeat(currentFeats, "pos-30", pos[i - 3], pos[i]);
            addFeat(currentFeats, "pos-3-2", pos[i - 3], pos[i - 2]);
        }

        if (i < tokens.length - 1) {
            addFeat(currentFeats, "w0+1", tokens[i], tokens[i + 1]);
            addFeat(currentFeats, "w+1", tokens[i + 1]);
            addFeat(currentFeats, "pos+1", pos[i + 1]);
            addFeat(currentFeats, "pos0+1", pos[i], pos[i + 1]);
        }

        if (i < tokens.length - 2) {
            addFeat(currentFeats, "w+2", tokens[i + 2]);
            addFeat(currentFeats, "pos+2", pos[i + 2]);
            addFeat(currentFeats, "pos0+2", pos[i], pos[i + 2]);
            addFeat(currentFeats, "pos+1+2", pos[i + 1], pos[i + 2]);
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
    public static List<Tuple> extractFeat(final SequenceTuple entry, boolean extractPrevTagFeat) {
        String[] tokens = entry.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[WORD_INDEX]).toArray(String[]::new);
        String[] pos = entry.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[POS_INDEX]).toArray(String[]::new);
        String[] tags = extractPrevTagFeat ? entry.entries.stream().map(t -> t.label).toArray(String[]::new) : null;

        List<List<String>> feats = IntStream.range(0, tokens.length)
                .mapToObj(i -> extractFeatSingle(i, tokens, pos, tags))
                .collect(Collectors.toList());

        List<Tuple> tuples = new ArrayList<>();

        for (int i = 0; i < feats.size(); i++) {
            List<String> singleTokenFeat = feats.get(i);
            LabeledVector v = new LabeledVector(singleTokenFeat.stream().toArray(String[]::new));

            Tuple t = new Tuple(v);
            t.label = entry.entries.get(i).label;
            tuples.add(t);
        }

        return tuples;
    }
}
