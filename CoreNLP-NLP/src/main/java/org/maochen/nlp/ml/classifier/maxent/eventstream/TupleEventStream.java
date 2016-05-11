package org.maochen.nlp.ml.classifier.maxent.eventstream;

import opennlp.model.Event;

import org.apache.commons.lang3.NotImplementedException;
import org.maochen.nlp.ml.Tuple;
import org.maochen.nlp.ml.vector.DenseVector;
import org.maochen.nlp.ml.vector.FeatNamedVector;
import org.maochen.nlp.ml.vector.SparseVector;
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

        String[] featName;
        float[] featVal = VectorUtils.doubleToFloat(tuple.vector.getVector());

        if (tuple.vector instanceof FeatNamedVector) {
            featName = ((FeatNamedVector) tuple.vector).featsName;
        } else if (tuple.vector instanceof SparseVector || tuple.vector instanceof DenseVector) {
            featName = new String[tuple.vector.getVector().length];
            for (int i = 0; i < featName.length; i++) {
                featName[i] = String.valueOf(i);
            }

        } else {
            throw new NotImplementedException("Unknown vector type");
        }

        return new Event(tuple.label, featName, featVal);
    }

    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    public TupleEventStream(List<Tuple> data) {
        this.dataIter = data.iterator();
    }

}
