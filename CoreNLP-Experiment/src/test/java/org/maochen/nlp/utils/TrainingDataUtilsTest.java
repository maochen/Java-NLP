package org.maochen.nlp.utils;

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
    public void test() {
        List<Tuple> originalList = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            originalList.add(new Tuple(1, null, "a"));
        }

        for (int i = 0; i < 200; i++) {
            originalList.add(new Tuple(1, null, "b"));
        }

        for (int i = 0; i < 320; i++) {
            originalList.add(new Tuple(1, null, "c"));
        }

        for (int i = 0; i < 2; i++) {
            originalList.add(new Tuple(1, null, "d"));
        }

        Collections.shuffle(originalList);
        List<Tuple> balanced = TrainingDataUtils.createBalancedTrainingData(originalList);
        assertEquals(562, originalList.size());
        assertEquals(2, balanced.stream().filter(x -> x.label.equals("a")).count());
        assertEquals(2, balanced.stream().filter(x -> x.label.equals("b")).count());
        assertEquals(2, balanced.stream().filter(x -> x.label.equals("c")).count());
        assertEquals(2, balanced.stream().filter(x -> x.label.equals("d")).count());
    }
}
