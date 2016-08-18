package org.maochen.nlp.ml.util;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 8/10/15.
 */
public class TrainingDataUtils {
    private static final Logger LOG = LoggerFactory.getLogger(TrainingDataUtils.class);

    /**
     * Regardless of the label, just consider isPosExample and !isPosExample
     *
     * @param trainingData
     * @return
     */
    public static List<Tuple> createBalancedTrainingDataForAll(final List<Tuple> trainingData) {
        List<Tuple> pos = trainingData.stream().filter(x -> x.isPosExample).collect(Collectors.toList());
        List<Tuple> neg = trainingData.stream().filter(x -> !x.isPosExample).collect(Collectors.toList());
        if (pos.size() < neg.size()) {
            neg = neg.subList(0, pos.size());
        } else if (pos.size() > neg.size()) {
            pos = pos.subList(0, neg.size());
        }

        pos.addAll(neg);
        return pos;
    }

    public static void reduceDimension(final List<Tuple> trainingData) {
        // FeatName, feat possible vals.
        // Do NOT store index for featname, index might not be same among tuples for the same featname.
        Map<String, Set<Double>> featNameValues = new HashMap<>();
        for (Tuple t : trainingData) {
            double[] val = t.vector.getVector();
            String[] name;
            if (t.vector instanceof FeatNamedVector) {
                name = ((FeatNamedVector) t.vector).featsName;
            } else {
                name = IntStream.range(0, t.vector.getVector().length).mapToObj(String::valueOf).toArray(String[]::new);
            }

            for (int i = 0; i < val.length; i++) {
                if (!featNameValues.containsKey(name[i])) {
                    featNameValues.put(name[i], new HashSet<>());
                }

                featNameValues.get(name[i]).add(val[i]);
            }
        }

        Set<String> singleValFeats = featNameValues.entrySet().stream().filter(x -> x.getValue().size() == 1).map(Map.Entry::getKey).collect(Collectors.toSet());

        LOG.debug("Single value feats: ");
        LOG.debug(singleValFeats.toString().replaceAll(", ", System.lineSeparator()));

        for (Tuple t : trainingData) {
            List<Double> featVal = new ArrayList<>();
            double[] originalVectorVal = t.vector.getVector();

            Set<Integer> indicesToBeRemoved;
            if (t.vector instanceof FeatNamedVector) {
                indicesToBeRemoved = new HashSet<>();
                String[] featName = ((FeatNamedVector) t.vector).featsName;

                for (int i = 0; i < featName.length; i++) {
                    if (singleValFeats.contains(featName[i])) {
                        indicesToBeRemoved.add(i);
                    }
                }
            } else {
                indicesToBeRemoved = singleValFeats.stream().map(Integer::parseInt).collect(Collectors.toSet());
            }


            for (int i = 0; i < originalVectorVal.length; i++) {
                if (!indicesToBeRemoved.contains(i)) {
                    featVal.add(originalVectorVal[i]);
                }
            }

            t.vector.setVector(featVal.stream().mapToDouble(x -> x).toArray());

            // Set feat name if needed
            if (t.vector instanceof FeatNamedVector) {
                List<String> featName = new ArrayList<>();
                String[] originalFeatName = ((FeatNamedVector) t.vector).featsName;

                for (int i = 0; i < originalVectorVal.length; i++) {
                    if (!indicesToBeRemoved.contains(i)) {
                        featName.add(originalFeatName[i]);
                    }
                }

                ((FeatNamedVector) t.vector).featsName = featName.stream().toArray(String[]::new);
            }
        }

    }

    /**
     * Balance the pos/neg data for every label.
     *
     * @param trainingData
     * @param cutoff       C(training data | Label) < cutoff won't be balanced.
     * @return
     */
    public static List<Tuple> createBalancedTrainingDataBasedOnLabel(final List<Tuple> trainingData, int cutoff) {
        LOG.debug("Original size:" + trainingData.size());
        Map<String, Long> tagCount = trainingData.parallelStream()
                .map(x -> new AbstractMap.SimpleImmutableEntry<>(x.label, 1))
                .collect(Collectors.groupingBy(Map.Entry::getKey, Collectors.counting()));

        LOG.debug(tagCount.toString());

        List<Tuple> copyTrainingData = new ArrayList<>();

        for (String tag : tagCount.keySet()) {
            List<Tuple> examples = trainingData.stream().filter(x -> tag.equals(x.label)).collect(Collectors.toList());
            if (examples.size() < cutoff) {
                LOG.debug("Keep all for " + tag + " -> " + tagCount.get(tag));
                copyTrainingData.addAll(examples);
                continue;
            }

            List<Tuple> posExample = examples.stream().filter(x -> x.isPosExample).collect(Collectors.toList());
            if (posExample.isEmpty()) {
                LOG.warn("Missing positive examples for: " + tag);
                copyTrainingData.addAll(examples);
                continue;
            }
            examples.removeAll(posExample);
            if (examples.isEmpty()) {
                LOG.warn("Missing negative examples for " + tag);
                copyTrainingData.addAll(examples);
                continue;
            }

            int minCount = Math.min(posExample.size(), examples.size());

            if (examples.size() == minCount) {
                posExample = posExample.subList(0, minCount);
            } else {
                examples = examples.subList(0, minCount);
            }

            copyTrainingData.addAll(posExample);
            copyTrainingData.addAll(examples);
        }

        LOG.debug("Size after balancing: " + trainingData.size());
        return copyTrainingData;
    }

    /**
     * Shuffle the data and split by proportion
     *
     * @param trainingData whole data collection.
     * @param proportion   scale from 0.1 - 1.0.
     * @return Left- small chunk. Right - large chunk.
     */
    public static Pair<List<Tuple>, List<Tuple>> splitData(final List<Tuple> trainingData, double proportion) {
        if (proportion < 0 || proportion > 1) {
            throw new RuntimeException("Proportion should between 0.0 - 1.0");
        }

        if (proportion > 0.5) {
            proportion = 1 - proportion;
        }

        List<Tuple> smallList = new ArrayList<>();
        List<Tuple> largeList = new ArrayList<>();

        int smallListSize = (int) Math.floor(proportion * trainingData.size());
        int ct = 0;

        Set<Integer> indices = new HashSet<>();
        while (ct < smallListSize && trainingData.size() > indices.size()) {
            int index = (int) (Math.random() * (trainingData.size() - 1));
            while (indices.contains(index)) {
                index = (int) (Math.random() * (trainingData.size() - 1));
            }
            indices.add(index);
            ct++;
        }

        smallList.addAll(indices.stream().map(trainingData::get).collect(Collectors.toList()));

        IntStream.range(0, trainingData.size())
                .filter(x -> !indices.contains(x))
                .forEach(i -> largeList.add(trainingData.get(i)));

        return new ImmutablePair<>(smallList, largeList);
    }


    public static List<SequenceTuple> readSeqFile(final InputStream trainingFile, final String delimiter, final int tagCol) {
        List<SequenceTuple> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new InputStreamReader(trainingFile))) {
            String line = br.readLine();

            int tupleId = 0;
            int seqId = 0;
            SequenceTuple sequenceTuple = new SequenceTuple();
            sequenceTuple.entries = new ArrayList<>();
            sequenceTuple.id = seqId;

            while (line != null) {
                if (line.trim().isEmpty()) {
                    data.add(sequenceTuple);
                    tupleId = 0;
                    seqId++;
                    sequenceTuple = new SequenceTuple();
                    sequenceTuple.entries = new ArrayList<>();
                    sequenceTuple.id = seqId;
                } else {
                    String[] fields = line.trim().split(delimiter);
                    String[] feats = IntStream.range(0, fields.length).filter(i -> i != tagCol).mapToObj(i -> fields[i]).toArray(String[]::new);
                    FeatNamedVector v = new FeatNamedVector(feats);
                    sequenceTuple.entries.add(new Tuple(tupleId++, v, fields[tagCol]));
                }
                line = br.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return data;
    }
}