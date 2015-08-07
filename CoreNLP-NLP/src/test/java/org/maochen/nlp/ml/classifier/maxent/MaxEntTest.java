package org.maochen.nlp.ml.classifier.maxent;

import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Maochen on 8/7/15.
 */
public class MaxEntTest {

    // tie=0.1899, lose=0.3686, win=0.4416
    private Map<String, Double> getMap(String s) {
        Map<String, Double> result = new HashMap<>();
        String[] tokens = s.split(",");
        for (String token : tokens) {
            String[] kv = token.split("=");
            String key = kv[0].trim();
            Double value = Double.parseDouble(kv[1].trim());
            result.put(key, value);
        }
        return result;
    }

    @Test
    public void testStringData() {
        List<String[]> traindata = new ArrayList<>();
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "win"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6666", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.3333", "win"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.6666", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.3333", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.75", "win"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.25", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.25", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "tie"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.25", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.25", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.25", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.6", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6666", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.4", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.7142", "win"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5714", "win"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.625", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.4285", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5714", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5555", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5555", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5555", "win"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.6", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5454", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.6", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4444", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.4545", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5454", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5384", "tie"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.4545", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5454", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5454", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5384", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5833", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.5714", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5384", "win"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5384", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.5384", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "tie"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5714", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5333", "win"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4666", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.625", "lose"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5333", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4375", "win"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6470", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5333", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5294", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4117", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6111", "tie"});
        traindata.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.5625", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5294", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4444", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6111", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5555", "win"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4736", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6315", "win"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5263", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.4736", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.55", "tie"});
        traindata.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.45", "win"});
        traindata.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.6190", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "tie"});
        traindata.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.55", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4285", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6363", "lose"});
        traindata.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5882", "lose"});
        traindata.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.5714", "lose"});
        traindata.add(new String[]{"away", "pdiff=0.9375", "ptwins=0.4545", "lose"});


        MaxEntClassifier maxent = new MaxEntClassifier();
        maxent.trainString(traindata);

//        maxent.persist(maxent.pathPrefix + "/maxentModel.txt");
//        maxent.model = null;
//        maxent.loadModel(maxent.pathPrefix + "/maxentModel.txt");

        List<String[]> predictData = new ArrayList<>();


        predictData.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5"});
        predictData.add(new String[]{"home", "pdiff=1.0625", "ptwins=0.5"});
        predictData.add(new String[]{"away", "pdiff=0.8125", "ptwins=0.5"});
        predictData.add(new String[]{"away", "pdiff=0.6875", "ptwins=0.6"});
        predictData.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.5"});
        predictData.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.3333"});
        predictData.add(new String[]{"away", "pdiff=1.0625", "ptwins=0.6666"});
        predictData.add(new String[]{"home", "pdiff=0.8125", "ptwins=0.6666"});
        predictData.add(new String[]{"home", "pdiff=0.9375", "ptwins=0.3333"});
        predictData.add(new String[]{"home", "pdiff=0.6875", "ptwins=0.5"});

        List<Map<String, Double>> predictExpected = new ArrayList<>(predictData.size());

        predictExpected.add(getMap("tie=0.1899, lose=0.3686, win=0.4416"));
        predictExpected.add(getMap("tie=0.2462, lose=0.3451, win=0.4087"));
        predictExpected.add(getMap("tie=0.2251, lose=0.5947, win=0.1802"));
        predictExpected.add(getMap("tie=0.1882, lose=0.6056, win=0.2062"));
        predictExpected.add(getMap("tie=0.2263, lose=0.3535, win=0.4202"));
        predictExpected.add(getMap("tie=0.2305, lose=0.3859, win=0.3836"));
        predictExpected.add(getMap("tie=0.2294, lose=0.5656, win=0.205"));
        predictExpected.add(getMap("tie=0.1688, lose=0.3409, win=0.4903"));
        predictExpected.add(getMap("tie=0.272, lose=0.3665, win=0.3614"));
        predictExpected.add(getMap("tie=0.1899, lose=0.3686, win=0.4416"));


        for (int i = 0; i < predictData.size(); i++) {
            String[] predict = predictData.get(i);
            Map<String, Double> actual = maxent.predict(predict);
            Map<String, Double> expected = predictExpected.get(i);
            for (String k : expected.keySet()) {
                assertTrue(actual.containsKey(k));
                assertEquals(expected.get(k), actual.get(k), 0.001);
            }
        }
    }
}
