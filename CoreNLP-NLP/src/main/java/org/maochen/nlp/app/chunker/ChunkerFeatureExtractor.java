package org.maochen.nlp.app.chunker;

import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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

    private static Map<String, String> BROWN_CLUSTER = new HashMap<>();

    private static int[] BROWN_PREFIX = new int[]{4, 6, 10, 20};

    static {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(ChunkerFeatureExtractor.class.getResourceAsStream("/brown.rcv1.3200.txt")))) {
            String line = br.readLine();

            while (line != null) {
                String[] fields = line.split("\\s");

                BROWN_CLUSTER.put(fields[1], fields[0]);
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // k: feat key - v: cluster Id
    private static Map<String, String> extractBrownFeat(String word) {
        if (!BROWN_CLUSTER.containsKey(word)) {
            return new HashMap<>();
        }

        String clusterId = BROWN_CLUSTER.get(word);

        return Arrays.stream(BROWN_PREFIX).mapToObj(p -> {
            int end = Math.min(p, clusterId.length());
            return new AbstractMap.SimpleEntry<>("brown_" + p, clusterId.substring(0, end));
        }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    }

    // Feats from http://www.aclweb.org/anthology/P10-1040
    public static List<String> extractFeatSingle(int i, final String[] tokens, final String[] pos) {
        List<String> currentFeats = new ArrayList<>();

        for (int index = Math.max(0, i - 2); index < Math.min(i + 3, tokens.length); index++) { // [-2,2]
            addFeat(currentFeats, "w" + (index - i), tokens[index]);
            addFeat(currentFeats, "pos" + (index - i), pos[index]);

            if (index == i - 1) {
                addFeat(currentFeats, "w-10", tokens[i - 1], tokens[i]);
                addFeat(currentFeats, "pos-10", pos[i - 1], pos[i]);
            } else if (index == i + 1) {
                addFeat(currentFeats, "w0+1", tokens[i], tokens[i + 1]);
                addFeat(currentFeats, "pos0+1", pos[i], pos[i + 1]);
            } else if (index == i - 2) {
                addFeat(currentFeats, "pos-2-1", pos[i - 2], pos[i - 1]);
                addFeat(currentFeats, "pos-2-10", pos[i - 2], pos[i - 1], pos[i]);
            } else if (index == i + 2) {
                addFeat(currentFeats, "pos+1+2", pos[i + 1], pos[i + 2]);
            }

            if (index == i - 1 && i < tokens.length - 1) {
                addFeat(currentFeats, "pos-10+1", pos[i - 1], pos[i], pos[i + 1]);
            }

            if (index == i + 2) {
                addFeat(currentFeats, "pos0+1+2", pos[i], pos[i + 1], pos[i + 2]);
            }
        }

        for (int index = Math.max(0, i - 2); index < Math.min(i + 3, tokens.length); index++) { // [-2,2]

            Map<String, String> feats = extractBrownFeat(tokens[index]);

            final int finalIndex = index;
            feats.entrySet().stream().forEach(entry -> {
                addFeat(currentFeats, entry.getKey() + "_" + (finalIndex - i), entry.getValue());
            });
        }


        return currentFeats;
    }

    /**
     * Single SequenceTuple featExtractor
     */
    public static List<Tuple> extractFeat(final SequenceTuple entry) {
        String[] tokens = entry.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[WORD_INDEX]).toArray(String[]::new);
        String[] pos = entry.entries.stream().map(tuple -> ((LabeledVector) tuple.vector).featsName[POS_INDEX]).toArray(String[]::new);

        List<List<String>> feats = IntStream.range(0, tokens.length)
                .mapToObj(i -> extractFeatSingle(i, tokens, pos))
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
