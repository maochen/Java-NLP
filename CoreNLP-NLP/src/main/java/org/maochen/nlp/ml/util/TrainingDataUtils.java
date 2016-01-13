package org.maochen.nlp.ml.util;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.maochen.nlp.ml.SequenceTuple;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 8/10/15.
 */
public class TrainingDataUtils {

    public static List<Tuple> createBalancedTrainingData(final List<Tuple> trainingData) {
        List<Tuple> copyTrainingData = new ArrayList<>(trainingData);
        Collections.shuffle(copyTrainingData);

        Map<String, Long> tagCount = trainingData.parallelStream()
                .map(x -> new AbstractMap.SimpleImmutableEntry<>(x.label, 1))
                .collect(Collectors.groupingBy(Map.Entry::getKey, Collectors.counting()));

        long minCount = tagCount.values().stream().min(Long::compareTo).get();

        Map<String, Integer> accumulateTagCount = tagCount.entrySet().stream()
                .map(Map.Entry::getKey)
                .map(x -> new AbstractMap.SimpleImmutableEntry<>(x, 0))
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));

        ListIterator<Tuple> iter = copyTrainingData.listIterator(copyTrainingData.size());
        while (iter.hasPrevious()) {
            Tuple tuple = iter.previous();
            int currentCount = accumulateTagCount.get(tuple.label);
            if (currentCount < minCount) {
                accumulateTagCount.put(tuple.label, currentCount + 1);
            } else {
                iter.remove();
            }
        }

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
                    LabeledVector v = new LabeledVector(feats);
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