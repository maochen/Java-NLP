package org.maochen.nlp.ml.classifier.maxent.eventstream;

import opennlp.model.Event;

import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.LabeledVector;
import org.maochen.nlp.util.VectorUtils;

import java.util.Iterator;
import java.util.List;

/**
 * Created by Maochen on 12/10/14.
 */
public class TupleEventStream implements EventStream {

    private Iterator<Tuple> dataIter;

    @Override
    public Event next() {
        Tuple tuple = dataIter.next();
        if (!(tuple.vector instanceof LabeledVector)) {
            throw new IllegalArgumentException("Please use LabeledVector to set feat label");
        }
        // Label, feature name, feature value
        return new Event(tuple.label, ((LabeledVector) tuple.vector).featsName,
                VectorUtils.doubleToFloat(tuple.vector.getVector()));
    }

    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    public TupleEventStream(List<Tuple> data) {
        this.dataIter = data.iterator();
    }

}
