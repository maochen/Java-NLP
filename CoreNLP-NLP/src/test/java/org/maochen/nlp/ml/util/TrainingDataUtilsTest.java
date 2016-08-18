package org.maochen.nlp.ml.util;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;
import org.maochen.nlp.ml.Tuple;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 8/10/15.
 */
public class TrainingDataUtilsTest {

    @Test
    public void testBalanceData() {
        List<Tuple> originalList = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            originalList.add(new Tuple(1, null, "a"));
            originalList.forEach(x -> x.isPosExample = true);
        }

        for (int i = 0; i < 2; i++) {
            Tuple t = new Tuple(1, null, "a");
            t.isPosExample = false;
            originalList.add(t);

        }

        Collections.shuffle(originalList);
        List<Tuple> balanced = TrainingDataUtils.createBalancedTrainingDataBasedOnLabel(originalList, 0);
        assertEquals(42, originalList.size());
        assertEquals(2, balanced.stream().filter(x -> x.label.equals("a")).filter(x -> x.isPosExample).count());
    }

    @Test
    public void splitDataTest() {
        List<Tuple> originalList = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            originalList.add(new Tuple(i, null, "Label"));
        }

        Pair<List<Tuple>, List<Tuple>> result = TrainingDataUtils.splitData(originalList, 0.6);

        assertEquals(400, originalList.size());
        assertEquals(160, result.getLeft().size());
        assertEquals(240, result.getRight().size());
    }
}
