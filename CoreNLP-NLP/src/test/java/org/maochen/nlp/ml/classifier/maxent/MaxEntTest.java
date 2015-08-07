package org.maochen.nlp.ml.classifier.maxent;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * TODO: Transform
 * Created by Maochen on 8/7/15.
 */
public class MaxEntTest {

    @Test
    public void test() {
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


        MaxEntClassifier maxent = new MaxEntClassifier(false);
        maxent.trainString(traindata);

//        maxent.persist(maxent.pathPrefix + "/maxentModel.txt");
//        maxent.model = null;
//        maxent.loadModel(maxent.pathPrefix + "/maxentModel.txt");

        List<String[]> predictData = new ArrayList<String[]>();
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

//        [home, pdiff=0.6875, ptwins=0.5]
//        {tie=0.1899, lose=0.3686, win=0.4416}
//        [home, pdiff=1.0625, ptwins=0.5]
//        {tie=0.2462, lose=0.3451, win=0.4087}
//        [away, pdiff=0.8125, ptwins=0.5]
//        {tie=0.2251, lose=0.5947, win=0.1802}
//        [away, pdiff=0.6875, ptwins=0.6]
//        {tie=0.1882, lose=0.6056, win=0.2062}
//        [home, pdiff=0.9375, ptwins=0.5]
//        {tie=0.2263, lose=0.3535, win=0.4202}
//        [home, pdiff=0.6875, ptwins=0.3333]
//        {tie=0.2305, lose=0.3859, win=0.3836}
//        [away, pdiff=1.0625, ptwins=0.6666]
//        {tie=0.2294, lose=0.5656, win=0.205}
//        [home, pdiff=0.8125, ptwins=0.6666]
//        {tie=0.1688, lose=0.3409, win=0.4903}
//        [home, pdiff=0.9375, ptwins=0.3333]
//        {tie=0.272, lose=0.3665, win=0.3614}
//        [home, pdiff=0.6875, ptwins=0.5]
//        {tie=0.1899, lose=0.3686, win=0.4416}
        for (String[] predict : predictData) {
            System.out.println(Arrays.toString(predict));
            System.out.println(maxent.predict(predict));
        }
    }
}
