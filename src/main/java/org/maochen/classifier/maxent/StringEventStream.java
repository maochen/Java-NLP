package org.maochen.classifier.maxent;

import opennlp.model.Event;
import opennlp.model.EventStream;
import opennlp.model.RealValueFileEventStream;
import org.apache.commons.lang3.StringUtils;

import java.util.Iterator;
import java.util.List;

/**
 * Created by Maochen on 12/10/14.
 */
public class StringEventStream implements EventStream {

    private Iterator<String[]> dataIter;

    private Event createEvent(String obs) {
        int lastSpace = obs.lastIndexOf(StringUtils.SPACE);
        if (lastSpace == -1)
            return null;
        else {
            String[] contexts = obs.substring(0, lastSpace).split("\\s+");
            float[] values = RealValueFileEventStream.parseContexts(contexts);
            // Label, feature name, feature value
            return new Event(obs.substring(lastSpace + 1), contexts, values);
        }
    }

    @Override
    public Event next() {
        String token = StringUtils.join(dataIter.next(), StringUtils.SPACE);
        return createEvent(token);
    }

    @Override
    public boolean hasNext() {
        return dataIter.hasNext();
    }

    public StringEventStream(List<String[]> data) {
        this.dataIter = data.iterator();
    }

}
