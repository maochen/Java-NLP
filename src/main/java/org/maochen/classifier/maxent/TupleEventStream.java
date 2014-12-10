package org.maochen.classifier.maxent;

import opennlp.model.Event;
import opennlp.model.EventStream;
import org.maochen.datastructure.Tuple;
import org.maochen.utils.VectorUtils;

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
        // Label, feature name, feature value
        return new Event(tuple.label, VectorUtils.intToString(tuple.featureVectorIndex), VectorUtils.doubleToFloat(tuple.featureVector));
    }

    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    public TupleEventStream(List<Tuple> data) {
        this.dataIter = data.iterator();
    }

}
