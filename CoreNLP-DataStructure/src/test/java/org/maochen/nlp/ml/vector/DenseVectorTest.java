package org.maochen.nlp.ml.vector;

import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Created by mguan on 6/28/16.
 */
public class DenseVectorTest {

    @Test
    public void testEquals() {
        DenseVector v1 = new DenseVector(new double[]{1, 3, 5});
        DenseVector v2 = new DenseVector(new double[]{1, 3, 5});
        assertEquals(v1, v2);
    }

    @Test
    public void testHashcode() {
        double[] primitive = new double[]{1, 3, 5};
        DenseVector v1 = new DenseVector(primitive);
        assertEquals(v1.hashCode(), Arrays.hashCode(primitive));
    }

    @Test
    public void testNotEqual() {
        DenseVector v1 = new DenseVector(new double[]{1, 5});
        DenseVector v2 = new DenseVector(new double[]{1, 3, 5});
        assertNotEquals(v1, v2);
    }
}
