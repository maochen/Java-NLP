package org.maochen.nlp.app.chunker;

import org.maochen.nlp.app.featextractor.BrownFeatExtractor;
import org.maochen.nlp.app.featextractor.IFeatureExtractor;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;

import java.util.ArrayList;
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
public class ChunkerFeatureExtractor implements IFeatureExtractor {

    public static final int WORD_INDEX = 0;
    public static final int POS_INDEX = 1;

    // Feats from http://www.aclweb.org/anthology/P10-1040
    public List<String> extractFeatSingle(int i, final String[] tokens, final String[] pos) {
        List<String> currentFeats = new ArrayList<>();

        for (int index = Math.max(0, i - 2); index < Math.min(i + 3, tokens.length); index++) { // [-2,2]
            IFeatureExtractor.addFeat(currentFeats, "w" + (index - i), tokens[index]);
            IFeatureExtractor.addFeat(currentFeats, "pos" + (index - i), pos[index]);

            if (index == i - 1) {
                IFeatureExtractor.addFeat(currentFeats, "w-10", tokens[i - 1], tokens[i]);
                IFeatureExtractor.addFeat(currentFeats, "pos-10", pos[i - 1], pos[i]);
            } else if (index == i + 1) {
                IFeatureExtractor.addFeat(currentFeats, "w0+1", tokens[i], tokens[i + 1]);
                IFeatureExtractor.addFeat(currentFeats, "pos0+1", pos[i], pos[i + 1]);
            } else if (index == i - 2) {
                IFeatureExtractor.addFeat(currentFeats, "pos-2-1", pos[i - 2], pos[i - 1]);
                IFeatureExtractor.addFeat(currentFeats, "pos-2-10", pos[i - 2], pos[i - 1], pos[i]);
            } else if (index == i + 2) {
                IFeatureExtractor.addFeat(currentFeats, "pos+1+2", pos[i + 1], pos[i + 2]);
            }

            if (index == i - 1 && i < tokens.length - 1) {
                IFeatureExtractor.addFeat(currentFeats, "pos-10+1", pos[i - 1], pos[i], pos[i + 1]);
            }

            if (index == i + 2) {
                IFeatureExtractor.addFeat(currentFeats, "pos0+1+2", pos[i], pos[i + 1], pos[i + 2]);
            }
        }

        currentFeats.addAll(BrownFeatExtractor.extractBrownFeat(i, -2, 2, tokens));
        return currentFeats;
    }

    /**
     * Single SequenceTuple featExtractor
     */
    @Override
    public List<Tuple> extractFeat(final SequenceTuple entry) {
        String[] tokens = entry.entries.stream().map(tuple -> ((FeatNamedVector) tuple.vector).featsName[WORD_INDEX]).toArray(String[]::new);
        String[] pos = entry.entries.stream().map(tuple -> ((FeatNamedVector) tuple.vector).featsName[POS_INDEX]).toArray(String[]::new);

        List<List<String>> feats = IntStream.range(0, tokens.length)
                .mapToObj(i -> extractFeatSingle(i, tokens, pos))
                .collect(Collectors.toList());

        List<Tuple> tuples = new ArrayList<>();

        for (int i = 0; i < feats.size(); i++) {
            List<String> singleTokenFeat = feats.get(i);
            FeatNamedVector v = new FeatNamedVector(singleTokenFeat.stream().toArray(String[]::new));

            Tuple t = new Tuple(v);
            t.label = entry.entries.get(i).label;
            tuples.add(t);
        }

        return tuples;
    }
}
