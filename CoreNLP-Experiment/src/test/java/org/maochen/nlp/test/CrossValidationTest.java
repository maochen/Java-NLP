package org.maochen.nlp.test;

import org.junit.Before;
import org.junit.Test;

import java.lang.reflect.InvocationTargetException;

import static org.junit.Assert.assertEquals;

/**
 * Created by Maochen on 8/11/15.
 */
public class CrossValidationTest {

    private CrossValidation crossValidation;


    @Before
    public void setUp() {
        crossValidation = new CrossValidation(1, null, false);
    }

//    private Object getField(Object instance, String fieldName) throws NoSuchFieldException,
//            ClassNotFoundException, IllegalAccessException {
//        Class cls = Class.forName("org.maochen.nlp.test.CrossValidation");
//        Field dataField = cls.getDeclaredField(fieldName);
//        dataField.setAccessible(true);
//        return dataField.get(instance);
//    }

    /*
       String label1 = "A";
        String label2 = "B";

        Class cls = Class.forName("org.maochen.nlp.test.CrossValidation");
        Constructor c = cls.getDeclaredConstructor(int.class, IClassifier.class, boolean.class);
        Object crossValidation = c.newInstance(1, null, false);

        List<Tuple> data = (List<Tuple>) getField(crossValidation, "data");
        Set<String> labels = (Set<String>) getField(crossValidation, "labels");

        data = new ArrayList<>();
        labels = ImmutableSet.of(label1, label2);

     */

    @Test
    public void testScoreCorrect() throws ClassNotFoundException, IllegalAccessException,
            InstantiationException, NoSuchFieldException, NoSuchMethodException,
            InvocationTargetException {

        CrossValidation.Score score = new CrossValidation.Score();
        score.trueNeg = 9760;
        score.truePos = 60;
        score.falseNeg = 40;
        score.falsePos = 140;

        assertEquals(0.3, score.getPrecision(), Double.MIN_NORMAL);
        assertEquals(0.6, score.getRecall(), Double.MIN_NORMAL);
        assertEquals(0.4, score.getF1(), Double.MIN_NORMAL);
        assertEquals(0.982, score.getAccurancy(), Double.MIN_NORMAL);
    }
}
