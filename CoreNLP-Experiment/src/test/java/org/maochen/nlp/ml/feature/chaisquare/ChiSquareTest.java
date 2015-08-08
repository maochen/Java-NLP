package org.maochen.nlp.ml.feature.chaisquare;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

/**
 * Created by Maochen on 8/7/15.
 */
public class ChiSquareTest {

    @Test
    public void testPValue() {
        double actual = ChiSquare.getPValue(16.2, 2);
        double expected = 3.03539138078901E-4;
        assertEquals(expected, actual, Double.MIN_NORMAL);

        actual = ChiSquare.getPValue(3, 1);
        expected = 0.08326451666354884;
        assertEquals(expected, actual, Double.MIN_NORMAL);
    }

    @Test
    public void testChiSquare1() {
        ChiSquare chiSquare = new ChiSquare();
        chiSquare.dataTable.put("Male", "Rep", 200);
        chiSquare.dataTable.put("Male", "Demo", 150);
        chiSquare.dataTable.put("Male", "Indep", 50);

        chiSquare.dataTable.put("Female", "Rep", 250);
        chiSquare.dataTable.put("Female", "Demo", 300);
        chiSquare.dataTable.put("Female", "Indep", 50);

        chiSquare.calculateChiSquare();
        
        assertEquals(1000, chiSquare.total);
        assertEquals(2, chiSquare.df);
        assertEquals(16.203703703703702, chiSquare.totalChiSquare, Double.MIN_NORMAL);
        assertEquals(3.0297754871455584E-4, chiSquare.totalPVal, Double.MIN_NORMAL);
        assertFalse(chiSquare.isFeatureUseful());
    }
}
