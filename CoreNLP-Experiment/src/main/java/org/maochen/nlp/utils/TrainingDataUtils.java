package org.maochen.nlp.utils;

import org.maochen.nlp.ml.Tuple;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by Maochen on 8/10/15.
 */
public class TrainingDataUtils {

    public static List<Tuple> createBalancedTrainingData(final List<Tuple> trainingData) {
        List<Tuple> copyTrainingData = new ArrayList<>(trainingData);
        Collections.shuffle(copyTrainingData);

        Map<String, Long> tagCount = trainingData.parallelStream()
                .map(x -> new AbstractMap.SimpleImmutableEntry<>(x.label, 1))
                .collect(Collectors.groupingBy(
                        AbstractMap.SimpleImmutableEntry::getKey, Collectors.counting()));

        long minCount = tagCount.values().stream().min(Long::compareTo).get();

        Map<String, Integer> accumulateTagCount = tagCount.entrySet().stream()
                .map(Map.Entry::getKey)
                .map(x -> new AbstractMap.SimpleImmutableEntry<>(x, 0))
                .collect(Collectors.toMap(AbstractMap.SimpleImmutableEntry::getKey,
                        AbstractMap.SimpleImmutableEntry::getValue));

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
}