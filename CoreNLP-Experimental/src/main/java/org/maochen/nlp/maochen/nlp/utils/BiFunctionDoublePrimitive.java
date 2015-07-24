package org.maochen.nlp.maochen.nlp.utils;

/**
 * Created by Maochen on 5/22/15.
 */
@FunctionalInterface
public interface BiFunctionDoublePrimitive {

    /**
     * Applies this function to the given arguments.
     *
     * @param t the first function argument
     * @param u the second function argument
     * @return the function result
     */
    double apply(double t, double u);
}
