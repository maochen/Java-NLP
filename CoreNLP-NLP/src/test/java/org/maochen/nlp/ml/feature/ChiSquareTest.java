package org.maochen.nlp.ml.feature;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

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
    }

    @Test
    public void testChiSquareWikiExample() {
        ChiSquare chiSquare = new ChiSquare();
        chiSquare.dataTable.put("Blue collar", "A", 90);
        chiSquare.dataTable.put("Blue collar", "B", 60);
        chiSquare.dataTable.put("Blue collar", "C", 104);
        chiSquare.dataTable.put("Blue collar", "D", 95);

        chiSquare.dataTable.put("White collar", "A", 30);
        chiSquare.dataTable.put("White collar", "B", 50);
        chiSquare.dataTable.put("White collar", "C", 51);
        chiSquare.dataTable.put("White collar", "D", 20);

        chiSquare.dataTable.put("Service", "A", 30);
        chiSquare.dataTable.put("Service", "B", 40);
        chiSquare.dataTable.put("Service", "C", 45);
        chiSquare.dataTable.put("Service", "D", 35);

        chiSquare.calculateChiSquare();

        assertEquals(650, chiSquare.total);
        assertEquals(6, chiSquare.df);
        assertEquals(24.571202858582602, chiSquare.totalChiSquare, Double.MIN_NORMAL);
        assertEquals(4.098425861096544E-4, chiSquare.totalPVal, Double.MIN_NORMAL);
    }
}
