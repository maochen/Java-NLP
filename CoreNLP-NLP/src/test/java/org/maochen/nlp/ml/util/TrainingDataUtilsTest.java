package org.maochen.nlp.ml.util;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.maochen.nlp.ml.vector.IVector;

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

    @Test
    public void reduceDimensionTest() {
        List<Tuple> tuples = new ArrayList<>();
        tuples.add(new Tuple(new double[]{1, 1, 3, 0}));
        tuples.add(new Tuple(new double[]{1, 0.4, 2, 0}));
        tuples.add(new Tuple(new double[]{1, 0.1, 2, 0}));
        tuples.add(new Tuple(new double[]{1, 0.9, 2, 0}));
        TrainingDataUtils.reduceDimension(tuples);

        for (Tuple t : tuples) {
            int vLen = t.vector.getVector().length;
            assertEquals(2, vLen);
        }
    }

    @Test
    public void reduceDimensionTest1() {
        List<Tuple> tuples = new ArrayList<>();
        tuples.add(new Tuple(new double[]{1, 0, 0, 0}));
        tuples.add(new Tuple(new double[]{0, 0.4, 0, 0}));
        tuples.add(new Tuple(new double[]{0, 0, 2, 0}));
        tuples.add(new Tuple(new double[]{0, 0, 0, 1}));
        TrainingDataUtils.reduceDimension(tuples);

        for (Tuple t : tuples) {
            int vLen = t.vector.getVector().length;
            assertEquals(4, vLen);
        }
    }

    @Test
    public void reduceDimensionTest2() {
        List<Tuple> tuples = new ArrayList<>();
        IVector v1 = new FeatNamedVector(new String[]{"a", "b", "c"});
        v1.setVector(new double[]{1, 0, 0});
        Tuple t1 = new Tuple(v1);
        tuples.add(t1);

        IVector v2 = new FeatNamedVector(new String[]{"d", "e", "f"});
        v2.setVector(new double[]{1, 0, 0});
        Tuple t2 = new Tuple(v2);
        tuples.add(t2);

        IVector v3 = new FeatNamedVector(new String[]{"a", "c", "e"});
        v3.setVector(new double[]{1, 0, 0});
        Tuple t3 = new Tuple(v3);
        tuples.add(t3);

        IVector v4 = new FeatNamedVector(new String[]{"b", "d", "f"});
        v4.setVector(new double[]{0, 1, 0});
        Tuple t4 = new Tuple(v4);
        tuples.add(t4);

        TrainingDataUtils.reduceDimension(tuples);

        for (Tuple t : tuples) {
            int vLen = t.vector.getVector().length;
            assertEquals(3, vLen);
        }
    }
}
