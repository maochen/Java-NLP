package org.maochen.nlp.ml.sampling;

import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.assertEquals;


/**
 * Created by Maochen on 10/17/15.
 */
public class ReservoirSamplingTest {

    private static List<Integer> list = IntStream.range(0, 100).mapToObj(x -> x).collect(Collectors.toList());

    @Test
    public void testAll() {
        Integer[] samples = ReservoirSampling.sample(100, list.iterator());
        for (int i = 0; i < 100; i++) {
            assertEquals((Integer) i, samples[i]);
        }
    }

    @Test
    public void testOne() {
        Integer[] samples = ReservoirSampling.sample(1, list.iterator());
        assertEquals(1, samples.length);
    }

    @Test
    public void testRandom() {
        Integer[] samples = ReservoirSampling.sample(12, list.iterator());
        assertEquals(12, samples.length);
    }

    @Test(expected = RuntimeException.class)
    public void testRuntimeException() {
        ReservoirSampling.sample(120, list.iterator());
    }

}
