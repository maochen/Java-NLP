package org.maochen.nlp.maochen.nlp.classifier.maxent;

import opennlp.model.Event;
import opennlp.model.EventStream;
import org.maochen.nlp.maochen.nlp.datastructure.Tuple;
import org.maochen.nlp.maochen.nlp.utils.VectorUtils;

import java.util.Iterator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by Maochen on 12/10/14.
 */
public class TupleEventStream implements EventStream {

    private Iterator<Tuple> dataIter;

    @Override
    public Event next() {
        Tuple tuple = dataIter.next();
        String[] featureName = IntStream.range(0, tuple.featureVector.length).mapToObj(String::valueOf).toArray(String[]::new);
        // Label, feature name, feature value
        return new Event(tuple.label, featureName, VectorUtils.doubleToFloat(tuple.featureVector));
    }

    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    public TupleEventStream(List<Tuple> data) {
        this.dataIter = data.iterator();
    }

}
