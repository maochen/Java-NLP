package org.maochen.nlp.ml.sampling;

import java.lang.reflect.Array;
import java.util.Iterator;
import java.util.Random;

/**
 * Created by Maochen on 10/17/15.
 */
public class ReservoirSampling {

    @SuppressWarnings("unchecked")
    public static <T> T[] sample(int k, Iterator<T> sequenceIter) {

        T firstElement = sequenceIter.next(); // For get the class from iter only.
        T[] samples = (T[]) Array.newInstance(firstElement.getClass(), k);

        for (int i = 0; i < k; i++) {
            if (i == 0) {
                samples[i] = firstElement;
                continue;
            }

            if (!sequenceIter.hasNext()) {
                throw new RuntimeException("Sequence has less elements than " + k);
            }
            samples[i] = sequenceIter.next();
        }

        for (int i = k + 1; sequenceIter.hasNext(); i++) {
            int j = new Random().nextInt(i); // original [1, i] make i inclusive, here, [0, i-1]
            T Si = sequenceIter.next();

            if (j < k) {
                samples[j] = Si;
            }
        }
        return samples;
    }
}
