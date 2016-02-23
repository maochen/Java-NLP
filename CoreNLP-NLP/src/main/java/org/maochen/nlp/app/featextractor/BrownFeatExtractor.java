package org.maochen.nlp.app.featextractor;

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

/**
 * Created by Maochen on 2/22/16.
 */
public class BrownFeatExtractor {

    private static Map<String, String> BROWN_CLUSTER = new HashMap<>();

    static {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(BrownFeatExtractor.class.getResourceAsStream("/brown.rcv1.3200.txt")))) {
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

    private static int[] BROWN_PREFIX = new int[]{4, 6, 10, 20};

    // k: feat key - v: cluster Id
    public static Map<String, String> extractBrownFeat(String word) {
        if (!BROWN_CLUSTER.containsKey(word)) {
            return new HashMap<>();
        }

        String clusterId = BROWN_CLUSTER.get(word);

        return Arrays.stream(BROWN_PREFIX).mapToObj(p -> {
            int end = Math.min(p, clusterId.length());
            return new AbstractMap.SimpleEntry<>("brown_" + p, clusterId.substring(0, end));
        }).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

    }


    public static List<String> extractBrownFeat(int currentIndex, int negOffset, int posOffset, String[] tokens) {

        List<String> currentFeats = new ArrayList<>();

        for (int index = Math.max(0, currentIndex + negOffset); index < Math.min(currentIndex + posOffset + 1, tokens.length); index++) {

            Map<String, String> feats = BrownFeatExtractor.extractBrownFeat(tokens[index]);

            final int finalIndex = index;
            feats.entrySet().stream().forEach(entry -> {
                IFeatureExtractor.addFeat(currentFeats, entry.getKey() + "_" + (finalIndex - currentIndex), entry.getValue());
            });
        }

        return currentFeats;
    }
}
